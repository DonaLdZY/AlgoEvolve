"""Keep the import identity required by models containing generated Python classes."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil


def _inference_function(tree, entrypoint):
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entrypoint), None)
    if entrypoint not in {'predict', 'solve', 'rollout'} or function is None:
        raise ValueError('Missing supported inference function')
    parameters = [a.arg for a in function.args.posonlyargs + function.args.args]
    if len(parameters) < 2 or any(v is None for v in function.args.kw_defaults):
        raise ValueError('Inference must accept model_path and data as positional arguments')
    if parameters[0] not in {'model_path', 'model'} or parameters[1] != 'data':
        raise ValueError('Inference positional arguments must be (model_path, data) or (model, data)')
    if len(parameters) - len(function.args.defaults) > 2:
        raise ValueError('Inference requires additional positional arguments')
    return function, parameters


def _body_nodes(function):
    # Nested helpers may use other inputs/outputs; do not attribute them to the entrypoint.
    pending = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def draft_delivery_contract(exported: Path) -> dict:
    """Extract source facts without executing candidate code or inventing types."""
    exported = Path(exported)
    manifest = json.loads((exported / "solution_manifest.json").read_text(encoding="utf-8"))
    source = exported / manifest.get("implementation_path", "solution.py")
    if not source.resolve().is_relative_to(exported.resolve()):
        raise ValueError("Implementation is outside the package")
    code = source.read_text(encoding="utf-8-sig")
    tree = ast.parse(code)
    entrypoint = manifest.get("entrypoint")
    function, parameters = _inference_function(tree, entrypoint)
    body = list(_body_nodes(function))
    data_name = parameters[1]
    fields = sorted({n.slice.value for n in body
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == data_name
        and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str) and isinstance(n.ctx, ast.Load)})
    referenced = {n.id for n in body if isinstance(n, ast.Name)}
    constants = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
    columns = next((constants[name] for name in ('PREDICT_COLUMNS', 'INPUT_COLUMNS', 'FEATURE_COLUMNS')
        if name in referenced and isinstance(constants.get(name), list) and constants[name]
        and all(isinstance(v, str) for v in constants[name])), [])
    returns = sorted({k.value for n in body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)
                      for k in n.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)})
    doc = ast.get_docstring(function) or ''
    output = ('返回对象字段：' + '、'.join(returns) + '。') if returns else '返回类型与结构需结合冻结实现和实际试算核对。'
    if function.returns:
        output += ' 源码返回注解：' + ast.unparse(function.returns) + '。'
    unresolved = ['字段业务含义、单位、合法范围和适用场景需要人工核对；静态分析不推断这些规则。',
                  '未自动生成可用于业务验收的输入样本；请用实际输入执行交付试算。']
    if not fields and not columns:
        unresolved.append('入口未直接声明完整输入结构，可能依赖辅助函数、嵌套对象或专用数据类型。')
    provenance = exported / 'contract_provenance.json'
    if provenance.is_file() and any(row.get('mode') == 'recorded_run_autorealize_link_snapshot'
                                  for row in json.loads(provenance.read_text(encoding='utf-8')).get('files', [])):
        unresolved.append('部分历史合同来自同次运行的 AutoRealize 链接，已保存本次整理快照；需人工核对是否与原实验合同一致。')
    draft = {'schema_version': 'autodecision.delivery.v1', 'status': 'draft', 'entrypoint': entrypoint,
        'signature': ast.unparse(function.args), 'implementation_path': source.relative_to(exported).as_posix(),
        'implementation_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'source_line': function.lineno,
        'description': doc or ('冻结选定算法的 ' + str(entrypoint) + ' 调用入口。'),
        'input_schema': {}, 'observed_input_fields': fields, 'dataframe_columns': columns,
        'dataframes': [{'pointer': '', 'columns': columns}] if columns else [],
        'input_description': ('入口引用的源码列清单：' + '、'.join(columns) + '。已提供 DataFrame 转换建议，须核对输入类型后使用 CSV 或 JSON 记录数组。') if columns
            else ('入口直接读取的字段：' + '、'.join(fields) + '；字段列表可能不完整。') if fields else '输入结构需核对冻结合同与实现。',
        'output_description': output, 'observed_output_fields': returns, 'unresolved': unresolved,
        'model_path': manifest.get('artifact_path'), 'source_node_id': manifest.get('node_id'),
        'verification_scope': '静态源码与包结构检查；尚不代表实际推理、业务验收或独立评估通过。'}
    (exported / 'delivery_contract.json').write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding='utf-8')
    (exported / 'DELIVERY.md').write_text('# 算法交付说明草稿\n\n' + draft['description'] + '\n\n'
        + '## 调用\n\n`solution.' + str(entrypoint) + '(model_path, data)`\n\n'
        + draft['input_description'] + '\n\n' + output + '\n\n## 待核对事项\n\n'
        + '\n'.join('- ' + item for item in unresolved) + '\n', encoding='utf-8')
    return draft


def package_inference_source(exported, source, module_name) -> None:
    """Wrap an evidence-identified source, preserving its bytes and module identity."""
    exported, source = Path(exported), Path(source)
    if not module_name.isidentifier():
        raise ValueError("Final model serialization module must be a Python identifier")
    manifest_path = exported / "solution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    # Validate the source contract before changing any wrapper files.
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    if not manifest.get('entrypoint'):
        available = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'predict', 'solve', 'rollout'}]
        if len(available) == 1:
            manifest['entrypoint'] = available[0]
    _inference_function(tree, manifest.get('entrypoint'))
    original = exported / "generated_solution.py"
    if not original.exists():
        shutil.copy2(exported / "solution.py", original)
    implementation = exported / (module_name + ".py")
    if source.resolve() != implementation.resolve():
        shutil.copy2(source, implementation)
    checksum = hashlib.sha256(implementation.read_bytes()).hexdigest()
    wrapper = f'''"""Portable entrypoints for the unchanged frozen generated implementation."""
from pathlib import Path
import hashlib
import importlib.util
import sys

_ROOT = Path(__file__).resolve().parent
_MODULE_NAME = {module_name!r}
_SOURCE = _ROOT / (_MODULE_NAME + ".py")
if hashlib.sha256(_SOURCE.read_bytes()).hexdigest() != {checksum!r}:
    raise RuntimeError("Packaged implementation changed after model export")
_implementation = sys.modules.get(_MODULE_NAME)
if _implementation is None:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    _implementation = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _implementation
    try:
        _spec.loader.exec_module(_implementation)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise

def __getattr__(name):
    return getattr(_implementation, name)

def _invoke_frozen(name, *args):
    # Models trained by standalone search scripts may refer to __main__.
    # Resolve those symbols from the frozen implementation without retraining
    # or rewriting the model; always restore the host application's module.
    previous = sys.modules.get('__main__')
    sys.modules['__main__'] = _implementation
    try:
        return getattr(_implementation, name)(*args)
    finally:
        if previous is None:
            sys.modules.pop('__main__', None)
        else:
            sys.modules['__main__'] = previous

def default_model_path():
    record = _ROOT / "model_path.txt"
    if not record.is_file():
        return None
    relative = record.read_text(encoding="utf-8").strip()
    if not relative:
        return None
    resolved = (_ROOT / relative).resolve()
    if not resolved.is_relative_to(_ROOT):
        raise RuntimeError("Model path must stay inside the delivery package")
    return resolved
'''
    signatures = {"predict": "model_path, data", "train": "data, artifact_dir",
                  "solve": "model_path, data", "rollout": "model_path, data", "train_policy": "data, artifact_dir"}
    for name in (manifest.get("entrypoint", "predict"), manifest.get("train_entrypoint", "train")):
        if name in signatures:
            args = signatures[name]
            wrapper += f"\ndef {name}({args}):\n    return _invoke_frozen({name!r}, {args})\n"
    (exported / "solution.py").write_text(wrapper, encoding="utf-8")
    manifest.update(implementation_path=implementation.name, generated_source_path=original.name,
                    serialization_module=module_name, portable_entrypoint_path="solution.py",
                    implementation_file_sha256=checksum)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    draft_delivery_contract(exported)


def package_final_solution(workspace, state) -> None:
    if state.get('status') not in {'completed', 'not_applicable', 'not_provided'}:
        return
    workspace = Path(workspace)
    exported = workspace / 'best_solution'
    source = workspace / 'final_evaluation/selected_solution.py'
    manifest_path = exported / 'solution_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('node_id') != state.get('selection', {}).get('node_id'):
            raise ValueError('Delivery manifest does not match the frozen selection')
    package_inference_source(exported, source, state.get('serialization_module', 'selected_final_solution'))
