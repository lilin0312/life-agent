"""
生活管家 AI-Agent 压力测试脚本
模拟 N 个并发用户同时访问，验证系统稳定性

用法:
    python test_stress.py                  # 默认5个用户
    python test_stress.py --users 10       # 10个并发用户
    python test_stress.py --users 5 --rounds 3   # 每个用户发3轮消息
"""
import asyncio
import argparse
import json
import os
import time
import statistics
import sys
from typing import Optional

try:
    import httpx
except ImportError:
    print("❌ 缅少 httpx，请运行: pip install httpx")
    sys.exit(1)

API_BASE = os.getenv("TEST_API_URL", "http://127.0.0.1:8000/api")

# 模拟用户的测试消息池
TEST_MESSAGES = [
    "你好，帮我算一下 1200 * 8 + 500",
    "现在几点了？",
    "帮我规划一个3天的北京旅行计划",
    "帮我记住我的生日是6月15号",
    "推荐一下今天中午吃什么",
    "帮我算一下 5000 * 0.2 - 300",
    "给我讲个笑话",
    "明天天气怎么样？",
    "帮我制定一个每周运动计划",
    "帮我算一下 (3000-800)/12",
    "推荐几部好看的电影",
    "怎么提高工作效率？",
    "帮我规划一下周末的活动",
    "今天适合吃什么？",
    "帮我算一下房租2500加上水电300等于多少",
]


class StressTester:
    """并发压力测试器"""

    def __init__(self, num_users: int = 5, rounds: int = 2, timeout: int = 35):
        self.num_users = num_users
        self.rounds = rounds
        self.timeout = timeout
        self.results: list[dict] = []
        self.errors: list[str] = []

    async def check_health(self) -> bool:
        """检查服务是否启动"""
        print("🔍 检查服务状态...")
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(f"{API_BASE}/health")
                data = resp.json()
                print(f"  状态: {data.get('status')}")
                print(f"  LLM:  {'✅' if data.get('llm_ready') else '❌'}")
                print(f"  记忆: {'✅' if data.get('memory_ready') else '❌'}")
                print(f"  RAG:  {'✅' if data.get('rag_ready') else '⚠️ 功能降级'}")
                return data.get("status") == "ok"
            except Exception as e:
                print(f"  ❌ 服务未启动: {e}")
                print(f"\n请先启动服务: python run.py")
                return False

    async def single_user(self, user_id: str, messages: list[str]) -> list[dict]:
        """模拟单个用户的完整交互流程"""
        user_results = []
        session_id = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i, msg in enumerate(messages):
                start = time.time()
                try:
                    resp = await client.post(
                        f"{API_BASE}/chat",
                        json={
                            "user_id": user_id,
                            "message": msg,
                            "session_id": session_id,
                        },
                    )
                    elapsed = time.time() - start

                    if resp.status_code == 200:
                        data = resp.json()
                        session_id = data.get("session_id")
                        user_results.append({
                            "user": user_id,
                            "round": i + 1,
                            "status": "✅",
                            "code": resp.status_code,
                            "time": round(elapsed, 2),
                            "tool": data.get("tool_used"),
                            "response_len": len(data.get("content", "")),
                        })
                    else:
                        user_results.append({
                            "user": user_id,
                            "round": i + 1,
                            "status": "❌",
                            "code": resp.status_code,
                            "time": round(elapsed, 2),
                            "error": resp.text[:100],
                        })

                except httpx.TimeoutException:
                    elapsed = time.time() - start
                    user_results.append({
                        "user": user_id,
                        "round": i + 1,
                        "status": "⏱️",
                        "time": round(elapsed, 2),
                        "error": "TIMEOUT",
                    })
                except Exception as e:
                    elapsed = time.time() - start
                    user_results.append({
                        "user": user_id,
                        "round": i + 1,
                        "status": "❌",
                        "time": round(elapsed, 2),
                        "error": str(e)[:100],
                    })

                # 模拟用户思考间隔（1-3秒）
                if i < len(messages) - 1:
                    await asyncio.sleep(1 + (hash(user_id + str(i)) % 20) / 10)

        return user_results

    async def run(self):
        """执行压力测试"""
        print(f"\n{'='*60}")
        print(f"  🚀 生活管家 AI-Agent 压力测试")
        print(f"  并发用户数: {self.num_users}")
        print(f"  每用户轮次: {self.rounds}")
        print(f"  超时阈值:   {self.timeout}s")
        print(f"{'='*60}\n")

        # 健康检查
        if not await self.check_health():
            return
        print()

        # 为每个用户分配消息
        user_tasks = []
        for i in range(self.num_users):
            uid = f"test_user_{i+1:03d}"
            # 从消息池中选取消息
            msgs = []
            for r in range(self.rounds):
                idx = (i * self.rounds + r) % len(TEST_MESSAGES)
                msgs.append(TEST_MESSAGES[idx])
            user_tasks.append(self.single_user(uid, msgs))

        # 并发执行
        print(f"⚡ 启动 {self.num_users} 个并发用户...\n")
        start_time = time.time()
        all_results = await asyncio.gather(*user_tasks)
        total_time = time.time() - start_time

        # 汇总结果
        self.results = [r for user_results in all_results for r in user_results]
        self.print_report(total_time)

    def print_report(self, total_time: float):
        """输出测试报告"""
        print(f"\n{'='*60}")
        print(f"  📊 压力测试报告")
        print(f"{'='*60}\n")

        # 详细结果
        print("  请求明细:")
        print(f"  {'用户':<16} {'轮次':<6} {'状态':<5} {'耗时':<8} {'工具':<10} {'备注'}")
        print(f"  {'-'*65}")
        for r in self.results:
            note = ""
            if r["status"] == "✅":
                note = f"回复{r.get('response_len',0)}字"
            elif r.get("error"):
                note = r["error"][:30]
            tool = r.get("tool", "-") or "-"
            print(f"  {r['user']:<16} {r['round']:<6} {r['status']:<5} {r['time']:<8.2f}s {tool:<10} {note}")

        # 统计分析
        success = [r for r in self.results if r["status"] == "✅"]
        failures = [r for r in self.results if r["status"] != "✅"]
        times = [r["time"] for r in success]

        print(f"\n  📈 统计:")
        print(f"  总请求数:   {len(self.results)}")
        print(f"  成功:       {len(success)} ✅")
        print(f"  失败/超时:  {len(failures)} ❌")
        print(f"  成功率:     {len(success)/len(self.results)*100:.1f}%")
        print(f"  总耗时:     {total_time:.1f}s")

        if times:
            print(f"\n  ⏱️  响应时间:")
            print(f"  最快:       {min(times):.2f}s")
            print(f"  最慢:       {max(times):.2f}s")
            print(f"  平均:       {statistics.mean(times):.2f}s")
            if len(times) >= 4:
                print(f"  P95:        {sorted(times)[int(len(times)*0.95)]:.2f}s")
                print(f"  P99:        {sorted(times)[min(int(len(times)*0.99), len(times)-1)]:.2f}s")

        # 工具使用统计
        tools_used = [r.get("tool") for r in success if r.get("tool")]
        if tools_used:
            print(f"\n  🔧 工具调用:")
            for tool in set(tools_used):
                count = tools_used.count(tool)
                print(f"  {tool}: {count}次")

        # 判定结果
        print(f"\n{'='*60}")
        passed = True
        issues = []

        if len(success) / len(self.results) < 0.95:
            passed = False
            issues.append(f"成功率 {len(success)/len(self.results)*100:.1f}% < 95%")

        if times and max(times) > 30:
            passed = False
            issues.append(f"最大响应时间 {max(times):.2f}s > 30s")

        if times and statistics.mean(times) > 20:
            passed = False
            issues.append(f"平均响应时间 {statistics.mean(times):.2f}s > 20s")

        if passed:
            print(f"  ✅ 测试通过！系统在 {self.num_users} 个并发用户下稳定运行")
        else:
            print(f"  ❌ 测试未通过：")
            for issue in issues:
                print(f"     - {issue}")

        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="生活管家 AI-Agent 压力测试")
    parser.add_argument("--users", type=int, default=5, help="并发用户数（默认5）")
    parser.add_argument("--rounds", type=int, default=2, help="每用户消息轮次（默认2）")
    parser.add_argument("--timeout", type=int, default=35, help="单请求超时秒数（默认35）")
    args = parser.parse_args()

    tester = StressTester(
        num_users=args.users,
        rounds=args.rounds,
        timeout=args.timeout,
    )
    asyncio.run(tester.run())


if __name__ == "__main__":
    main()
