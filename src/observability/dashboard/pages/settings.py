"""
设置页面。

配置系统设置。
"""

import streamlit as st


def render():
    """渲染设置页面"""
    st.title("⚙️ 系统设置")
    st.info("🚧 此页面正在开发中（任务 G6）")
    st.markdown("""
    ### 计划功能：
    - 编辑 LLM 配置
    - 编辑 Embedding 配置
    - 编辑检索参数
    - 保存和重新加载设置
    - 验证配置更改
    """)


if __name__ == "__main__":
    render()
