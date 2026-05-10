"""
查询测试页面。

测试查询并查看结果。
"""

import streamlit as st


def render():
    """渲染查询测试页面"""
    st.title("🔍 查询测试")
    st.info("🚧 此页面正在开发中（任务 G5）")
    st.markdown("""
    ### 计划功能：
    - 交互式查询输入
    - 查看检索到的 Chunk 和分数
    - 显示多模态结果（文本 + 图片）
    - 调整检索参数
    - 比较不同查询策略
    """)


if __name__ == "__main__":
    render()
