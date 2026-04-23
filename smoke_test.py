"""快速冒烟测试：验证各模块能正确导入和串联"""
from data_generator import RigorousMetasurfaceSimulator
from forward_model import train_forward_model, ForwardPredictor
from inverse_design import TandemTrainer, InverseDesigner
from metasurface_design import design_anomalous_refraction_array
print('所有模块导入成功')

sim = RigorousMetasurfaceSimulator()
X, Y = sim.generate_dataset(n_samples=200)
fwd, hist, scaler = train_forward_model(X, Y, epochs=5, verbose=False)
tandem = TandemTrainer(fwd, scaler)
tandem.train_with_progress(epochs=5, verbose=False)
res = design_anomalous_refraction_array(
    tandem.inverse_model, scaler, fwd,
    wavelength=1550e-9, period=600e-9, n_elements=5
)
print(f'端到端测试通过，实现折射角: {res["actual_angle"]:.1f}°')
