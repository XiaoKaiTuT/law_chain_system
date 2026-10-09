"""
函数功能：评估脚本的启动引导 —— 在 import 任何项目模块【之前】导入

解决的问题：
    项目的 base/config.py 只在【进程环境变量】里查找 DEEPSEEK_API_KEY：
        self.DASHSCOPE_API_KEY = os.getenv('DEEPSEEK_API_KEY', ...)
    fallback 是占位符 'YOUR_DEEPSEEK_API_KEY'。

    而 Windows 用户变量（在“系统属性 → 环境变量”里设的那种）
    只对设置之后【新启动】的进程生效。于是一个常见现象是：
        - PyCharm 里能跑通（PyCharm 是后启动的）
        - 命令行/脚本里报 401（进程环境快照里没有这个变量）

    评估脚本必须走真实链路，而真实链路里的 LLM 客户端读的是
    base/config.py，因此必须在它被 import 之前把 key 补进进程环境变量。

做法：
    import 本模块 → 若 os.environ 里没有 DEEPSEEK_API_KEY，
    则从 Windows 用户变量（注册表）读出来并【写入 os.environ】。
    之后再 import base.config，就能拿到正确的 key。

为什么不能只靠 wrappers.resolve_api_key()：
    那个函数只服务评估模块自己的裁判 LLM；
    项目自己的 llm_client 读取的是 base/config.py 的快照，两者互不相通。

用法：
    在任何评估入口脚本的最顶部：
        import rag_qa.evaluation.bootstrap  # noqa: F401
    或直接：
        from rag_qa.evaluation import bootstrap
"""

import os

# 已知的占位符值（与 base/config.py 的 fallback 保持一致）
_PLACEHOLDER = "YOUR_DEEPSEEK_API_KEY"


def _read_windows_user_env(name: str) -> str | None:
    """从 Windows 用户变量（注册表 HKCU:\\Environment）读取变量值"""
    if os.name != "nt":
        return None
    try:
        import subprocess

        result = subprocess.run(
            ["reg", "query", r"HKCU\Environment", "/v", name],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="ignore",
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == name and parts[1].startswith("REG_"):
                return " ".join(parts[2:]).strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def ensure_api_key_in_environ() -> str:
    """
    函数功能：确保 os.environ 里有可用的 DEEPSEEK_API_KEY

    优先级：
        ① 进程环境变量已有且不是占位符 → 保持不动
        ② Windows 用户变量 → 写入 os.environ
        ③ 都没有 → 保持原样（后续会由 wrappers 给出可操作的报错）

    :return: 状态描述，便于脚本打印
    """
    current = os.getenv("DEEPSEEK_API_KEY")
    if current and current != _PLACEHOLDER:
        return "进程环境变量已提供，无需处理"

    from_user_var = _read_windows_user_env("DEEPSEEK_API_KEY")
    if from_user_var and from_user_var != _PLACEHOLDER:
        os.environ["DEEPSEEK_API_KEY"] = from_user_var
        return f"已从 Windows 用户变量注入（长度 {len(from_user_var)}）"

    return "⚠️ 未找到可用的 DEEPSEEK_API_KEY（项目 LLM 调用会失败）"


STATUS = ensure_api_key_in_environ()

if __name__ == "__main__":
    print(f"bootstrap 状态：{STATUS}")
    print(f"os.environ['DEEPSEEK_API_KEY'] 是否存在："
          f"{bool(os.getenv('DEEPSEEK_API_KEY'))}")
