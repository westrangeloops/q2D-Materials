from ..utils.file_handlers import vasp_load, save_vasp

class q2D_analysis:
    def __init__(self, B, X, crystal):
        self.path = '/'.join(crystal.split('/')[0:-1])
        self.name = crystal.split('/')[-1]
        self.B = B
        self.X = X
        self.perovskite_df, self.box = vasp_load(crystal)
    
    def isolate_spacer(self, order=None):
        crystal_df = self.perovskite_df
        B = self.B
        # Find the planes of the perovskite.
        b_planes = crystal_df.query("Element == @B").sort_values(by='Z')
        b_planes['Z'] = b_planes['Z'].apply(lambda x: round(x, 1))
        b_planes.drop_duplicates(subset='Z', inplace=True)
        b_planes.reset_index(inplace=True, drop=True)

        if len(b_planes.values) > 1:
            b_planes['Diff'] = b_planes['Z'].diff()
            id_diff_max = b_planes['Diff'].idxmax()
            b_down_plane = b_planes.iloc[id_diff_max - 1:id_diff_max] 
            b_up_plane = b_planes.iloc[id_diff_max:id_diff_max + 1]
            b_down_plane = b_down_plane['Z'].values[0] + 1
            b_up_plane = b_up_plane['Z'].values[0] - 1
            # Now lets create a df with only the elements that are between that!
            # We call that the salt
            iso_df = crystal_df.query('Z <= @b_up_plane and Z >= @b_down_plane ')
        
        elif len(b_planes.values) == 1:
            b_unique_plane = b_planes['Z'].values[0] + 1
            iso_df = crystal_df.query('Z >= @b_unique_plane')

        else:
            print('There was not found {} planes'.format(B))
            return None

        # Update the box to be 10 amstrong in Z
        try:
            iso_df.loc[:, 'Z'] = iso_df['Z'] - iso_df['Z'].min()
            box = self.box
            box[0][2][2] = iso_df['Z'].sort_values(ascending=False).iloc[0] + 10
            # Save the system
            name = self.path + '/' + 'salt_' + self.name
            print('Your isolated salt file was save as: ', name)
            save_vasp(iso_df, box, name, order=order)

        except Exception as e:
            print(f"Error creating the SALT: {e}") 