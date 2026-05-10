"""
E2E 冒烟测试：Dashboard 页面

目标：
- 验证 Dashboard 各页面模块可以导入
- 确保没有 Python 语法错误
- 验证基本结构完整性

测试策略：
1. 测试 Dashboard 应用文件存在
2. 测试所有页面模块文件存在
3. 测试页面模块有 render 函数
4. 验证没有语法错误

注意：由于 Dashboard 依赖大量外部服务和配置，完整的渲染测试需要真实环境。
本测试主要验证代码结构和基本完整性。
"""

import pytest
from pathlib import Path
import ast


@pytest.fixture
def dashboard_app_path():
    """返回 Dashboard 应用路径"""
    return Path(__file__).parent.parent.parent / "src" / "observability" / "dashboard" / "app.py"


@pytest.fixture
def dashboard_pages_dir():
    """返回 Dashboard 页面目录"""
    return Path(__file__).parent.parent.parent / "src" / "observability" / "dashboard" / "pages"


def test_dashboard_app_exists(dashboard_app_path):
    """测试：Dashboard 应用文件存在"""
    assert dashboard_app_path.exists(), f"Dashboard 应用文件不存在: {dashboard_app_path}"


def test_dashboard_app_no_syntax_errors(dashboard_app_path):
    """测试：Dashboard 代码没有语法错误"""
    try:
        with open(dashboard_app_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Dashboard 代码有语法错误: {e}")


def test_dashboard_pages_directory_exists(dashboard_pages_dir):
    """测试：Dashboard 页面目录存在"""
    assert dashboard_pages_dir.exists(), f"页面目录不存在: {dashboard_pages_dir}"
    assert dashboard_pages_dir.is_dir(), f"页面路径不是目录: {dashboard_pages_dir}"


def test_dashboard_pages_init_exists(dashboard_pages_dir):
    """测试：页面模块 __init__.py 存在"""
    init_file = dashboard_pages_dir / "__init__.py"
    assert init_file.exists(), f"__init__.py 不存在: {init_file}"


def test_dashboard_page_files_exist(dashboard_pages_dir):
    """测试：所有页面文件存在"""
    expected_pages = [
        "overview.py",
        "traces.py",
        "data_browser.py",
        "ingestion.py",
        "ingestion_traces.py",
        "query_traces.py",
        "query_test.py",
        "settings.py",
        "evaluation_panel.py",
    ]

    missing_pages = []
    for page_file in expected_pages:
        page_path = dashboard_pages_dir / page_file
        if not page_path.exists():
            missing_pages.append(page_file)

    assert not missing_pages, f"缺少页面文件: {', '.join(missing_pages)}"


def test_dashboard_pages_no_syntax_errors(dashboard_pages_dir):
    """测试：所有页面文件没有语法错误"""
    page_files = list(dashboard_pages_dir.glob("*.py"))
    page_files = [f for f in page_files if f.name != "__init__.py"]

    syntax_errors = []
    for page_file in page_files:
        try:
            with open(page_file, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code)
        except SyntaxError as e:
            syntax_errors.append(f"{page_file.name}: {e}")

    assert not syntax_errors, f"页面文件有语法错误:\n" + "\n".join(syntax_errors)


def test_dashboard_pages_have_render_function(dashboard_pages_dir):
    """测试：所有页面文件包含 render 函数定义"""
    page_files = list(dashboard_pages_dir.glob("*.py"))
    page_files = [f for f in page_files if f.name != "__init__.py"]

    missing_render = []
    for page_file in page_files:
        try:
            with open(page_file, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code)

            # 查找 render 函数定义
            has_render = any(
                isinstance(node, ast.FunctionDef) and node.name == "render"
                for node in ast.walk(tree)
            )

            if not has_render:
                missing_render.append(page_file.name)

        except Exception as e:
            pytest.fail(f"无法解析 {page_file.name}: {e}")

    assert not missing_render, f"以下页面缺少 render 函数: {', '.join(missing_render)}"


def test_dashboard_app_imports_pages(dashboard_app_path):
    """测试：Dashboard app 导入页面模块"""
    with open(dashboard_app_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 验证导入语句存在
    assert "from src.observability.dashboard.pages import" in code, \
        "Dashboard app 应该导入页面模块"


def test_dashboard_app_has_main_function(dashboard_app_path):
    """测试：Dashboard app 有 main 函数"""
    with open(dashboard_app_path, 'r', encoding='utf-8') as f:
        code = f.read()
    tree = ast.parse(code)

    has_main = any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in ast.walk(tree)
    )

    assert has_main, "Dashboard app 应该有 main 函数"


def test_dashboard_app_structure():
    """测试：Dashboard 应用结构完整"""
    dashboard_dir = Path(__file__).parent.parent.parent / "src" / "observability" / "dashboard"

    # 验证目录结构
    assert (dashboard_dir / "app.py").exists(), "缺少 app.py"
    assert (dashboard_dir / "pages").exists(), "缺少 pages 目录"
    assert (dashboard_dir / "pages" / "__init__.py").exists(), "缺少 pages/__init__.py"


def test_dashboard_pages_count(dashboard_pages_dir):
    """测试：Dashboard 页面数量符合预期"""
    page_files = list(dashboard_pages_dir.glob("*.py"))
    page_files = [f for f in page_files if f.name != "__init__.py"]

    # 至少应该有 9 个页面
    assert len(page_files) >= 9, f"页面数量不足，当前: {len(page_files)}, 预期: >= 9"


def test_dashboard_services_directory():
    """测试：Dashboard services 目录存在"""
    services_dir = Path(__file__).parent.parent.parent / "src" / "observability" / "dashboard" / "services"

    assert services_dir.exists(), "services 目录应该存在"
    assert services_dir.is_dir(), "services 应该是目录"


def test_dashboard_pages_importable(dashboard_pages_dir):
    """测试：页面模块文件存在（不实际导入）"""
    page_modules = [
        "overview",
        "traces",
        "data_browser",
        "ingestion",
        "ingestion_traces",
        "query_traces",
        "query_test",
        "settings",
        "evaluation_panel",
    ]

    missing_modules = []
    for module_name in page_modules:
        module_file = dashboard_pages_dir / f"{module_name}.py"
        if not module_file.exists():
            missing_modules.append(module_name)

    assert not missing_modules, f"缺少页面模块文件: {', '.join(missing_modules)}"


def test_dashboard_app_entry_point(dashboard_app_path):
    """测试：Dashboard 有正确的入口点"""
    with open(dashboard_app_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 验证有 if __name__ == "__main__" 入口
    assert 'if __name__ == "__main__"' in code, \
        "Dashboard app 应该有 __main__ 入口点"


def test_dashboard_page_consistency(dashboard_pages_dir):
    """测试：所有页面文件命名一致"""
    page_files = list(dashboard_pages_dir.glob("*.py"))
    page_files = [f for f in page_files if f.name != "__init__.py"]

    # 验证文件名都是小写加下划线
    invalid_names = []
    for page_file in page_files:
        name = page_file.stem
        if not name.replace("_", "").islower():
            invalid_names.append(page_file.name)

    assert not invalid_names, f"页面文件命名不符合规范（应该是小写+下划线）: {', '.join(invalid_names)}"


def test_dashboard_comprehensive_smoke(dashboard_pages_dir):
    """测试：Dashboard 综合冒烟测试"""
    dashboard_dir = dashboard_pages_dir.parent

    # 1. 验证主要文件存在
    assert (dashboard_dir / "app.py").exists()
    assert (dashboard_pages_dir / "__init__.py").exists()

    # 2. 验证页面文件存在
    page_files = list(dashboard_pages_dir.glob("*.py"))
    assert len(page_files) >= 10  # 至少 9 个页面 + __init__.py

    # 3. 验证没有明显的语法错误
    for page_file in page_files:
        try:
            with open(page_file, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
        except SyntaxError:
            pytest.fail(f"{page_file.name} 有语法错误")

    # 4. 验证所有页面都有 render 函数
    for page_file in page_files:
        if page_file.name == "__init__.py":
            continue

        with open(page_file, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)

        has_render = any(
            isinstance(node, ast.FunctionDef) and node.name == "render"
            for node in ast.walk(tree)
        )

        assert has_render, f"{page_file.name} 缺少 render 函数"
