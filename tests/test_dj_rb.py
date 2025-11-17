from q2D_Materials.core.creator import q2D_creator
from ase.io import write

creator = q2D_creator()

dj_rb = creator.create_perovskite('DJ',
    A_ions='Sr', B_ions='Nb', X_ions='O',
    spacer='Rb',
    supercell=[1, 1, 2]
)

write('dj_rb_sr.vasp', dj_rb)

dj_rb = creator.create_perovskite('DJ',
    A_ions='Ca', B_ions='Ta', X_ions='O',
    spacer='Rb',
    supercell=[1, 1, 2]
)

write('dj_rb_ca.vasp', dj_rb)