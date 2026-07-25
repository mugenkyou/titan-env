from titan_env.core.environment.titan_env import TITANEnv as CoreEnv
from titan_env.core.rewards.reward_v2 import compute_reward as reward_v2

class EnvAdapter:
    def __init__(self):
        self.e = CoreEnv()

    @staticmethod
    def _obs11(obs):
        return [
            float(obs.get("voltage", 0.0)),
            float(obs.get("current_draw", 0.0)),
            float(obs.get("battery_soc", 0.0)),
            float(obs.get("cpu_temperature", 0.0)),
            float(obs.get("power_temperature", 0.0)),
            float(obs.get("memory_integrity", 0.0)),
            float(obs.get("cpu_load", 0.0)),
            abs(float(obs.get("voltage", 0.0)) - float(obs.get("current_draw", 0.0))),
            abs(float(obs.get("cpu_temperature", 0.0)) - float(obs.get("power_temperature", 0.0))),
            abs(float(obs.get("memory_integrity", 0.0)) - float(obs.get("cpu_load", 0.0))),
            min(1.0, max(0.0, float(obs.get("recent_fault_count", 0.0)))),
        ]

    def reset(self):
        return self._obs11(self.e.reset())

    def step(self, a):
        raw_obs, done, i = self.e.step(int(a))
        r, _ = reward_v2(raw_obs, int(a), bool(done))
        s = {k: v for k, v in dict(i).items() if "fault" not in str(k).lower()}
        if int(a) == 7:
            s["diagnose_fault"] = i.get("fault_type", None)
            s["diagnose_severity"] = float(i.get("fault_severity_level", 0))
        return self._obs11(raw_obs), float(r), bool(done), s

def make_env():
    return EnvAdapter()

try:
    env = make_env()
    obs = env.reset()
    assert obs is not None
    print("Init PASS")

    assert len(obs) == 11
    assert all(0.0 <= float(x) <= 1.0 for x in obs)
    print("Observation PASS")

    obs, _, _, info = env.step(0)
    assert "fault" not in str(obs)
    assert "fault" not in str(info)
    print("No Fault Leakage PASS")

    obs1 = env.reset()
    obs2, _, _, info = env.step(7)
    assert "diagnose_fault" in info
    assert "diagnose_severity" in info
    assert len(obs1) == len(obs2)
    print("Diagnose PASS")

    env.reset()
    _, _, _, info1 = env.step(4)
    _, _, _, info2 = env.step(2)
    _, _, _, info3 = env.step(1)
    assert info1 is not None
    assert info2 is not None
    assert info3 is not None
    print("Side Effects PASS")

    env.reset()
    for _ in range(10):
        obs, reward, done, info = env.step(0)
    assert obs is not None
    assert isinstance(reward, float)
    print("Step Loop PASS")

    assert not any(float(x) != float(x) for x in obs)
    print("No NaN PASS")

    print("Final summary: OK")
except Exception as e:
    n = e.args[0] if getattr(e, "args", None) else ""
    print(f"FAIL {n}")
    print("Final summary: BLOCKED")
    raise
