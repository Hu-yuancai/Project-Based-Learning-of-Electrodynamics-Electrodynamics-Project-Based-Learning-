"""
测试脚本：验证新物理模型在正式重写前是否正确

测试内容：
1. 数据生成器：相位覆盖范围、透射率范围
2. 正向网络：输入/输出维度、sin/cos 编码
3. Tandem 训练：梯度流、损失下降
4. 阵列设计：相位梯度计算

运行方式：python test_physics.py
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. 测试物理模型（数据生成器）
# ============================================================

def test_physics_model():
    print("=" * 60)
    print("TEST 1: 物理模型（Rytov + Fabry-Perot）")
    print("=" * 60)

    lambda0 = 1550e-9
    k0 = 2 * np.pi / lambda0
    n_Si = 3.4777
    n_sub = 1.444
    n_air = 1.0
    H = 900e-9

    def effective_index(w, p):
        f = w / p
        n_TE2 = f * n_Si**2 + (1 - f) * n_air**2
        delta_n = (n_Si**2 - n_air**2) / 3 * (f * (1 - f))**2
        n_eff_sq = n_TE2 + delta_n * (k0 * p)**2
        return np.sqrt(max(n_eff_sq, 1.0))

    def compute_transmission(w, p):
        n_eff = effective_index(w, p)
        phi_prop = n_eff * k0 * H
        r1 = (n_air - n_eff) / (n_air + n_eff)
        r2 = (n_eff - n_sub) / (n_eff + n_sub)
        exp_term = np.exp(1j * phi_prop)
        t = (1 - r1) * (1 - r2) * exp_term / (1 - r1 * r2 * np.exp(2j * phi_prop))
        T = min(abs(t)**2, 1.0)
        phi_deg = np.rad2deg(np.angle(t))
        return T, phi_deg

    # 扫描 w，固定 p=600nm，看相位变化
    p_test = 600e-9
    w_vals = np.linspace(80e-9, 500e-9, 50)
    phases = []
    transmissions = []
    for w in w_vals:
        f = w / p_test
        if 0.1 <= f <= 0.8:
            T, phi = compute_transmission(w, p_test)
            phases.append(phi)
            transmissions.append(T)

    phases = np.array(phases)
    transmissions = np.array(transmissions)

    print(f"  固定 p=600nm，扫描 w=80~500nm:")
    print(f"  相位范围: [{phases.min():.1f}°, {phases.max():.1f}°]  (期望覆盖 >300°)")
    print(f"  透射率范围: [{transmissions.min():.3f}, {transmissions.max():.3f}]  (期望 >0.3)")
    phase_span = phases.max() - phases.min()
    assert phase_span > 200, f"相位覆盖不足！只有 {phase_span:.1f}°，期望 >200°"
    assert transmissions.mean() > 0.3, f"透射率太低！均值 {transmissions.mean():.3f}"
    print("  [PASS] 相位覆盖和透射率正常")

    # 测试不同 p 下的相位变化
    print(f"\n  扫描不同 (w, p) 组合的相位分布:")
    all_phases = []
    for p in np.linspace(400e-9, 1000e-9, 10):
        for w in np.linspace(80e-9, 500e-9, 20):
            f = w / p
            if 0.1 <= f <= 0.8 and p < lambda0 / n_sub:
                T, phi = compute_transmission(w, p)
                all_phases.append(phi)

    all_phases = np.array(all_phases)
    print(f"  全局相位范围: [{all_phases.min():.1f}°, {all_phases.max():.1f}°]")
    print(f"  样本数: {len(all_phases)}")
    assert all_phases.max() - all_phases.min() > 300, "相位覆盖不足 300°"
    print("  [PASS] 全局相位覆盖 >300°")
    return True


# ============================================================
# 2. 测试正向网络（输入 2 维，输出 sin/cos）
# ============================================================

class ForwardNet(nn.Module):
    """新版正向网络：(w,p) -> (sin_phi, cos_phi, T)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 3)  # [sin_phi, cos_phi, T_logit]
        )

    def forward(self, x):
        out = self.net(x)
        sin_phi = out[:, 0]
        cos_phi = out[:, 1]
        T = torch.sigmoid(out[:, 2])
        return sin_phi, cos_phi, T


def test_forward_network():
    print("\n" + "=" * 60)
    print("TEST 2: 正向网络（2D 输入，sin/cos 输出）")
    print("=" * 60)

    model = ForwardNet()
    batch = torch.randn(32, 2)
    sin_phi, cos_phi, T = model(batch)

    assert sin_phi.shape == (32,), f"sin_phi shape 错误: {sin_phi.shape}"
    assert cos_phi.shape == (32,), f"cos_phi shape 错误: {cos_phi.shape}"
    assert T.shape == (32,), f"T shape 错误: {T.shape}"
    assert T.min() >= 0 and T.max() <= 1, f"T 超出 [0,1]: [{T.min():.3f}, {T.max():.3f}]"
    print(f"  输出形状: sin_phi={sin_phi.shape}, cos_phi={cos_phi.shape}, T={T.shape}")
    print(f"  T 范围: [{T.min().item():.3f}, {T.max().item():.3f}]  (期望在 [0,1])")
    print("  [PASS] 正向网络结构正确")

    # 测试 sin/cos 损失函数
    true_phi_deg = torch.tensor([0., 90., 180., -90., -180.])
    true_phi_rad = torch.deg2rad(true_phi_deg)
    true_sin = torch.sin(true_phi_rad)
    true_cos = torch.cos(true_phi_rad)

    pred_sin = true_sin + 0.1 * torch.randn_like(true_sin)
    pred_cos = true_cos + 0.1 * torch.randn_like(true_cos)

    mse_loss = nn.MSELoss()
    phase_loss = mse_loss(pred_sin, true_sin) + mse_loss(pred_cos, true_cos)
    norm_loss = torch.mean((pred_sin**2 + pred_cos**2 - 1)**2)
    total = phase_loss + 0.1 * norm_loss

    assert total.item() > 0, "损失为 0，可能有问题"
    print(f"  sin/cos 损失测试: phase_loss={phase_loss.item():.4f}, norm_loss={norm_loss.item():.4f}")
    print("  [PASS] 损失函数计算正确")

    # 测试从 sin/cos 恢复相位
    # atan2 返回 (-π, π]，±180° 在浮点下可能互换，用周期性误差衡量
    recovered = torch.rad2deg(torch.atan2(true_sin, true_cos))
    diff = torch.abs(recovered - true_phi_deg)
    # 考虑周期性：误差取 diff 和 360-diff 的最小值
    periodic_err = torch.min(diff, 360.0 - diff)
    max_err = torch.max(periodic_err).item()
    assert max_err < 0.01, f"相位恢复误差过大: {max_err:.4f}°"
    print(f"  atan2 相位恢复最大误差（周期性）: {max_err:.6f}°")
    print("  [PASS] sin/cos -> 相位 恢复正确")
    return True


# ============================================================
# 3. 测试 Tandem 梯度流
# ============================================================

class InverseNet(nn.Module):
    """逆向网络：sin_phi, cos_phi -> (w_norm, p_norm)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def test_tandem_gradient_flow():
    print("\n" + "=" * 60)
    print("TEST 3: Tandem 梯度流")
    print("=" * 60)

    forward_net = ForwardNet()
    inverse_net = InverseNet()

    # 冻结正向网络
    for p in forward_net.parameters():
        p.requires_grad = False
    forward_net.eval()

    optimizer = torch.optim.Adam(inverse_net.parameters(), lr=1e-3)

    # 检查正向网络参数确实被冻结
    for p in forward_net.parameters():
        assert not p.requires_grad, "正向网络参数未被冻结！"
    print("  正向网络参数已冻结")

    # 模拟一步训练
    target_phi_deg = torch.rand(32, 1) * 360 - 180
    target_phi_rad = torch.deg2rad(target_phi_deg)
    target_sin = torch.sin(target_phi_rad)
    target_cos = torch.cos(target_phi_rad)
    target_input = torch.cat([target_sin, target_cos], dim=1)

    optimizer.zero_grad()

    # 逆向预测 (w_norm, p_norm)
    geo_norm = inverse_net(target_input)  # [batch, 2]

    # 反归一化到物理范围（nm），然后标准化给正向网络
    w_nm = geo_norm[:, 0] * (500 - 80) + 80   # [80, 500] nm
    p_nm = geo_norm[:, 1] * (1073 - 400) + 400  # [400, 1073] nm

    # 标准化（模拟 StandardScaler）
    w_mean, w_std = 290.0, 120.0
    p_mean, p_std = 736.5, 195.0
    w_scaled = (w_nm - w_mean) / w_std
    p_scaled = (p_nm - p_mean) / p_std
    geo_scaled = torch.stack([w_scaled, p_scaled], dim=1)

    # 正向网络预测
    pred_sin, pred_cos, _ = forward_net(geo_scaled)

    # 损失：在相位空间
    loss = nn.MSELoss()(pred_sin, target_sin.squeeze()) + \
           nn.MSELoss()(pred_cos, target_cos.squeeze())

    loss.backward()

    # 检查逆向网络有梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in inverse_net.parameters())
    assert has_grad, "逆向网络没有梯度！梯度流断了"
    print(f"  Tandem 损失: {loss.item():.4f}")
    print("  逆向网络有梯度: True")

    # 检查正向网络没有梯度
    no_fwd_grad = all(p.grad is None for p in forward_net.parameters())
    assert no_fwd_grad, "正向网络不应该有梯度！"
    print("  正向网络无梯度（冻结正确）: True")

    optimizer.step()
    print("  [PASS] Tandem 梯度流正常")
    return True


# ============================================================
# 4. 测试数据集生成（小规模）
# ============================================================

def test_dataset_generation():
    print("\n" + "=" * 60)
    print("TEST 4: 数据集生成（100 样本快速测试）")
    print("=" * 60)

    lambda0 = 1550e-9
    k0 = 2 * np.pi / lambda0
    n_Si = 3.4777
    n_sub = 1.444
    n_air = 1.0
    H = 900e-9

    W_MIN, W_MAX = 80e-9, 500e-9
    P_MIN, P_MAX = 400e-9, 1073e-9

    def effective_index(w, p):
        f = w / p
        n_TE2 = f * n_Si**2 + (1 - f) * n_air**2
        delta_n = (n_Si**2 - n_air**2) / 3 * (f * (1 - f))**2
        n_eff_sq = n_TE2 + delta_n * (k0 * p)**2
        return np.sqrt(max(n_eff_sq, 1.0))

    def compute_transmission(w, p):
        n_eff = effective_index(w, p)
        phi_prop = n_eff * k0 * H
        r1 = (n_air - n_eff) / (n_air + n_eff)
        r2 = (n_eff - n_sub) / (n_eff + n_sub)
        t = (1 - r1) * (1 - r2) * np.exp(1j * phi_prop) / \
            (1 - r1 * r2 * np.exp(2j * phi_prop))
        T = min(abs(t)**2, 1.0)
        phi_deg = np.rad2deg(np.angle(t))
        return T, phi_deg

    np.random.seed(42)
    n = 200
    w_raw = np.random.uniform(W_MIN, W_MAX, n)
    p_raw = np.random.uniform(P_MIN, P_MAX, n)

    valid_w, valid_p, valid_T, valid_phi = [], [], [], []
    for w, p in zip(w_raw, p_raw):
        f = w / p
        if 0.1 <= f <= 0.8 and p < lambda0 / n_sub:
            T, phi = compute_transmission(w, p)
            if not np.isnan(T) and not np.isnan(phi):
                valid_w.append(w * 1e9)
                valid_p.append(p * 1e9)
                valid_T.append(T)
                valid_phi.append(phi)

    valid_phi = np.array(valid_phi)
    valid_T = np.array(valid_T)

    print(f"  有效样本: {len(valid_w)}/{n}  (有效率 {len(valid_w)/n*100:.1f}%)")
    print(f"  相位范围: [{valid_phi.min():.1f}°, {valid_phi.max():.1f}°]")
    print(f"  透射率范围: [{valid_T.min():.3f}, {valid_T.max():.3f}]")

    assert len(valid_w) > n * 0.5, f"有效率太低: {len(valid_w)/n*100:.1f}%"
    assert valid_phi.max() - valid_phi.min() > 200, \
        f"相位覆盖不足: {valid_phi.max() - valid_phi.min():.1f}°"
    assert valid_T.mean() > 0.3, f"透射率均值太低: {valid_T.mean():.3f}"
    print("  [PASS] 数据集生成正常")
    return True


# ============================================================
# 5. 测试阵列相位梯度计算
# ============================================================

def test_array_phase_gradient():
    print("\n" + "=" * 60)
    print("TEST 5: 阵列相位梯度（广义斯涅尔定律）")
    print("=" * 60)

    lambda0 = 1550e-9
    theta_t = 30.0  # 目标折射角
    n_elements = 21

    # 用平均周期 p_avg 计算相位梯度
    # 实际设计中每个单元的 p 不同，这里用固定 p 验证公式
    p_avg = 600e-9
    k0 = 2 * np.pi / lambda0

    phase_gradient = k0 * np.sin(np.deg2rad(theta_t))
    positions = np.arange(n_elements) * p_avg
    ideal_phases_rad = phase_gradient * positions
    ideal_phases_deg = np.rad2deg(ideal_phases_rad) % 360
    ideal_phases_deg = (ideal_phases_deg + 180) % 360 - 180

    print(f"  目标折射角: {theta_t}°")
    print(f"  相位梯度: {phase_gradient:.4f} rad/m = {np.rad2deg(phase_gradient)*1e-6:.4f} °/μm")
    print(f"  相位范围: [{ideal_phases_deg.min():.1f}°, {ideal_phases_deg.max():.1f}°]")

    # 验证：从相位梯度反推折射角
    phase_rad = np.deg2rad(ideal_phases_deg)
    unwrapped = np.unwrap(phase_rad)
    coeffs = np.polyfit(positions, unwrapped, 1)
    gradient_measured = coeffs[0]
    sin_theta = gradient_measured / k0
    angle_recovered = np.rad2deg(np.arcsin(np.clip(sin_theta, -1, 1)))

    print(f"  反推折射角: {angle_recovered:.2f}°  (误差: {abs(angle_recovered - theta_t):.2f}°)")
    assert abs(angle_recovered - theta_t) < 1.0, \
        f"相位梯度计算误差过大: {abs(angle_recovered - theta_t):.2f}°"
    print("  [PASS] 相位梯度计算正确")
    return True


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  超构表面逆向设计系统 - 物理模型测试")
    print("=" * 60)

    results = {}
    tests = [
        ("物理模型", test_physics_model),
        ("正向网络", test_forward_network),
        ("Tandem梯度流", test_tandem_gradient_flow),
        ("数据集生成", test_dataset_generation),
        ("阵列相位梯度", test_array_phase_gradient),
    ]

    for name, fn in tests:
        try:
            fn()
            results[name] = "PASS"
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            results[name] = f"FAIL: {e}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = f"ERROR: {e}"

    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    all_pass = True
    for name, result in results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status:4s}  {name}: {result if result != 'PASS' else 'OK'}")

    print()
    if all_pass:
        print("  所有测试通过，可以开始正式重写各模块。")
    else:
        print("  有测试失败，请先修复再继续。")
