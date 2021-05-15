from ansys.mapdl.core import launch_mapdl
import numpy as np

mapdl = launch_mapdl()
print(mapdl)

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
