import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt


model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# === Paramètres de ton planeur (À MESURER) ===
RHO = 1.225          # densité air
CD0 = 0.02           # traînée de friction plaque
S_WING = 0.126       # surface aile (m²)   <-- MESURE
S_TAIL = 0.06        # surface empennage   <-- MESURE
AR = 2.0             # allongement aile = envergure²/surface

# IDs des bodies (vérifie avec le listing)
WING = model.body("centralunit").id        # <-- ADAPTE
TAIL = model.body("part_1_2").id   # <-- ADAPTE

# Coefficients plaque plane (formules vues ensemble)
def aero_coeffs(alpha):
    CL = 2*np.sin(alpha)*np.cos(alpha) / (1 + 2/AR)   # correction allongement
    CD = CD0 + 2*np.sin(alpha)**3
    return CL, CD

def apply_surface(body_id, S, extra_alpha=0.0):
    """Applique portance+traînée sur une surface, à son origine."""
    R = data.xmat[body_id].reshape(3,3)      # rotation corps→monde
    # vitesse du body dans le monde (au centre de masse du body)
    v = data.cvel[body_id][3:] + np.cross(data.cvel[body_id][:3],
                                            -data.xpos[body_id] + data.xipos[body_id])
    V = np.linalg.norm(v)
    if V < 0.3:
        return
    v_body = R.T @ v                          # vitesse dans le repère du corps
    # le corps avance selon son axe X ; incidence = angle entre vitesse et X
    alpha = -np.arctan2(v_body[2], v_body[0]) + extra_alpha
    CL, CD = aero_coeffs(alpha)
    # directions dans le monde : portance ⊥ vitesse, traînée ∥ vitesse
    v_hat = v / V
    lift_dir = np.cross(v_hat, [0, 1, 0])    # ⊥ vitesse, dans le plan vertical
    lift_dir /= np.linalg.norm(lift_dir) + 1e-9
    F = 0.5 * RHO * V**2 * S * (CL * lift_dir - CD * v_hat)
    # force appliquée au centre de masse du body (pas de couple parasite)
    data.xfrc_applied[body_id, :3] += F

# === Lancement ===
data.qpos[2] = 100.0     # altitude
data.qvel[0] = 1.0      # vitesse de lancement

drawt = []
drawz = []
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 30:
        data.xfrc_applied[:] = 0                     # reset des forces du pas précédent
        # incidence de l'empennage : le flap dévie le flux
        flap_left  = data.qpos[model.jnt_qposadr[model.joint("left_motor").id]]   # angle du flap (rad)
        flap_right = data.qpos[model.jnt_qposadr[model.joint("right_motor").id]]
        apply_surface(WING, S_WING)
        apply_surface(TAIL, S_TAIL, extra_alpha=(flap_left + flap_right) / 2)
        mujoco.mj_step(model, data)
        viewer.cam.lookat[:] = data.qpos[:3]   # regarde le planeur
        z = data.qpos[2]
        if z <= 0.0:
            print(f"Touché le sol : t={data.time:.2f}s, distance={data.qpos[0]:.1f}m")
            break
        print(f"t={data.time:5.2f}  z={z:6.2f}  x={data.qpos[0]:7.2f}  V={np.linalg.norm(data.qvel[:3]):.2f}")
        drawt.append(data.time)
        drawz.append(z)

plt.plot(drawt,drawz)
plt.show()
