def interface_reciever(ase_cell):
    """
    This recieves an ase object from the creator and constructs new objects from it depending if its bulk, DJ, RP, or monolayer.
    """
    if ase_cell.get_chemical_formula() == 'bulk':
        return 
    elif ase_cell.get_chemical_formula() == 'DJ':
        return NaN
    elif ase_cell.get_chemical_formula() == 'RP':
        return None #TODO: Implement RP interface
    elif ase_cell.get_chemical_formula() == 'monolayer':
        return None #TODO: Implement monolayer interface


def create_bulk_interface(ase_cell):
    """
    This creates a bulk interface from an ase object.
    """
    