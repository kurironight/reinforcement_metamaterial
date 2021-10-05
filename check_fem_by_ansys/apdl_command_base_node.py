from ansys.mapdl.core import launch_mapdl
import numpy as np
import time

mapdl = launch_mapdl()
print(mapdl)
start = time.time()

mapdl.finish()
mapdl.clear()
mapdl.prep7()
# 節点部分の設定
mapdl.k(1, 0, 0)
mapdl.k(2, 5, 0)
mapdl.k(3, 5, 5)

# エッジ部分の設定
mapdl.l(1, 2, 1)
mapdl.l(2, 3, 1)

# 材料物性値の設定
mapdl.et(1, 3)
mapdl.mp("ex", 1, 1)  # ヤング率
mapdl.mp("prxy", 1, 0.3)  # ポアソン比

# エッジの太さ，指定
mapdl.r(1, 2, 0.6667, 2, 0)
mapdl.r(2, 4, 5.3334, 4, 0)
mapdl.mat(1)
mapdl.real(1)
mapdl.lmesh(1)
mapdl.real(2)
mapdl.lmesh(2)

mapdl.finish()

mapdl.run('/solu')
mapdl.antype('static')

mapdl.dk(1, "all", 0)
mapdl.fk(3, "FX", 0)
mapdl.fk(3, "FY", -1000)
mapdl.solve()
mapdl.finish()

x_disp = mapdl.post_processing.nodal_displacement('X')
y_disp = mapdl.post_processing.nodal_displacement('Y')
z_rot = mapdl.post_processing.nodal_rotation('Z')

displacement = np.stack([x_disp, y_disp, z_rot]).T.flatten()
print(displacement)


# grab the result from the ``mapdl`` instance
result = mapdl.result
nnum, nodal_stress = result.nodal_stress(0)
print(nodal_stress)
"""
#result.plot_nodal_stress(0, 'Y', cmap='bwr')

nnum, principal_nodal_stress = result.principal_nodal_stress(0)
#result.plot_principal_nodal_stress(0, 'S1', cmap='bwr')
print(principal_nodal_stress)
# von_mises = principal_nodal_stress[:, -1]  # von-Mises stress is the right most column
# print(von_mises)
# Must use nanmax as stress is not computed at mid-side nodes
#max_stress = np.nanmax(von_mises)
# print(max_stress)

node1_stress = nodal_stress[0]
rhox = node1_stress[0]
rhoy = node1_stress[1]
tauxy = node1_stress[3]

rho1 = (rhox + rhoy) / 2 + np.sqrt(((rhox - rhoy) / 2)**2 + tauxy**2)
print(rho1)

rho2 = (rhox + rhoy) / 2 - np.sqrt(((rhox - rhoy) / 2)**2 + tauxy**2)
print(rho2)

mises = np.sqrt((rho1**2 + rho2**2 + (rho2 - rho1)**2) / 2)
print(mises)
"""
