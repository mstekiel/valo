from dataclasses import dataclass
import numpy as np
from pathlib import Path



import lmfit
from lmfit import Parameters

# import spinwaves
from spinwaves import Atom, Crystal, SpinW, Coupling
from spinwaves.symmetry.magnetic_symmetry import MSG
from spinwaves.utils.linalg import DMI

from spinw_studio import plot_structure  # renders in valo; see README.md


HOME_PATH = Path(__file__).parent


def load_system(parameters: Parameters, show_struct: bool=False, silent=True) -> SpinW:
    Fz = parameters['Fz'].value
    atoms = [Atom(label='Fe', r=(0,   0.5, 0),   m=(-1,0,Fz), s=2.5)]
    Pbnm = MSG.from_xyz_strings(generators=[
        'x+1/2,-y+1/2,-z, -1',
        '-x,-y,-z, +1',
        '-x,-y,z+1/2, +1',
    ])

    cs = Crystal(lattice_parameters=[5.3, 5.6, 7.5, 90, 90, 90], 
                 atoms=atoms, MSG=Pbnm)
    
    # print(cs)

    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }


    ### Extract the model parameters
    # Negative couplings are FM, positive are AF
    Ka = parameters['Ka'].value*2
    Kc = parameters['Kc'].value*2
    # Kac = parameters['Kac'].value

    J1ab = parameters['J1ab'].value
    J1c  = parameters['J1c'].value
    J2a = parameters['J2a'].value
    J2b = parameters['J2b'].value
    J2d1  = parameters['J2d1'].value
    J2d2  = parameters['J2d2'].value

    Dab_x = parameters['Dab_x'].value
    Dab_y = parameters['Dab_y'].value
    Dab_z = parameters['Dab_z'].value
    
    Dc_x = parameters['Dc_x'].value
    Dc_y = parameters['Dc_y'].value
    Dc_z = parameters['Dc_z'].value

    # J6 = parameters["J6"].value
    # J7 = parameters["J7"].value
    J8 = parameters["J8"].value
    J9a = parameters["J9a"].value
    J9b = parameters["J9b"].value
    # J10 = parameters["J10"].value
    # J11a = parameters["J11a"].value
    # J11b = parameters["J11b"].value
    # J12a = parameters["J12a"].value
    # J12b = parameters["J12b"].value
    # J13 = parameters["J13"].value


    couplings = []
    
    # Wrap up the couplings in one list
    # Single-ion anisotropies
    # K = np.array([
    #     [ Ka, 0, Kac],
    #     [  0, 0, 0],
    #     [Kac, 0, Kc],
    # ])
    couplings += [Coupling(label=f'K_Fe', n_uvw=[0,0,0], id1=0, id2=0, J=np.diag([Ka,0,Kc]))]
    # Nearest neighbour along the c axis
    DMI_c = DMI([Dc_x,Dc_y, Dc_z])
    couplings += [Coupling(label=f'J1c', n_uvw=[0,0,0], id1=0, id2=1, J=J1c*np.eye(3,3)+DMI_c)]


    # Nearest-neighbour in-plane exchanges. Thay also contain the Dzialoszynski-Moriya interaction


    # The DMI schemeis from Park et al 2018 picture
    # One issue with that picture is it seems D32 and D32' are interexchanged, based on arrow directions.
    # Maybe this can be tested
    DMI_ab = DMI([Dab_x, Dab_y, Dab_z])
    couplings += [Coupling(label=f'J1a', n_uvw=[0,0,0], id1=0, id2=2, J=J1ab*np.eye(3,3)+DMI_ab)]

    # Exchange between atoms along main orthorhombic directions
    couplings += [Coupling(label=f'J2a', n_uvw=[1,0,0], id1=0, id2=0, J=J2a*np.eye(3,3))]
    couplings += [Coupling(label=f'J2b', n_uvw=[0,1,0], id1=0, id2=0, J=J2b*np.eye(3,3))]

    couplings += [
        Coupling(label=f'J2d1', n_uvw=[0,0,0], id1=0, id2=3, J=J2d1*np.eye(3,3)),
        Coupling(label=f'J2d2', n_uvw=[-1,0,0], id1=0, id2=3, J=J2d2*np.eye(3,3))
    ]

    # Further couplings
    couplings += [
        # Coupling(label="J6", n_uvw=[-1,0,-1], id1=0, id2=1,     J=J6*np.eye(3,3)),
        # Coupling(label="J7", n_uvw=[0,-1,-1], id1=0, id2=1,     J=J7*np.eye(3,3)),
        Coupling(label="J8", n_uvw=[0,0,-1], id1=0, id2=0,      J=J8*np.eye(3,3)),
        Coupling(label="J9a", n_uvw=[-1,1,0], id1=0, id2=0 ,    J=J9a*np.eye(3,3)),
        Coupling(label="J9b", n_uvw=[-1,-1,0], id1=0, id2=0,    J=J9b*np.eye(3,3)),
        # Coupling(label="J10", n_uvw=[-2,0,0], id1=0, id2=2 ,    J=J10*np.eye(3,3)),
        # Coupling(label="J11a", n_uvw=[-1,0,-1], id1=0, id2=2,   J=J11a*np.eye(3,3)),
        # Coupling(label="J11b", n_uvw=[-1,0,1], id1=0, id2=2,    J=J11b*np.eye(3,3)),
        # Coupling(label="J12a", n_uvw=[-1,-1,-1], id1=0, id2=1,  J=J12a*np.eye(3,3)),
        # Coupling(label="J12b", n_uvw=[-1,1,-1], id1=0, id2=1,   J=J12b*np.eye(3,3)),
        # Coupling(label="J13", n_uvw=[-1,-1,0], id1=0, id2=2,    J=J13*np.eye(3,3)),
    ]

    if not silent:
        print('Provided couplings')
        for cpl in couplings:
            print(cpl)

    # Construct the main object that is able to determine excitation spectrum
    sw = SpinW(crystal=cs, 
               couplings=couplings,
               magnetic_modulation=magnetic_modulation)
    
    if not silent:
        print('Symmetrized couplings')
        for cpl in sw.couplings_all:
            print(cpl)
    
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = {
            'boundaries': ([-0.1, 1.1],[-0.1,1.1],[-0.1,1.02]),   # uvw extent
            'atom_colors': {'Fe': 'Orange'},
            'show_moments': True,
            'coupling_colors': {                                    # kept for when couplings are drawn
                'J1c': 'Orange', 
                'J1ab':'Gray', 
                'J2a':'Green', 
                'J2b':'Red', 
                'J2d':'Blue'
                }
            }
        # Blocks until the window is closed; returns the plot_options as you left them (paste them back above).
        plot_options = plot_structure(sw, plot_options=plot_opts)
        if not silent:
            print('plot_options when the window was closed:', plot_options)
    
    return sw


if __name__ == "__main__":
    # PLACEHOLDER values (all couplings 0): enough to look at the structure. Put your fitted parameters here.
    #     uv run --group spinw python examples/spinw_studio/lfo_spinw.py
    params = Parameters()
    for name in ['Fz', 'Ka', 'Kc', 'J1ab', 'J1c', 'J2a', 'J2b', 'J2d1', 'J2d2', 'Dab_x', 'Dab_y', 'Dab_z',
                 'Dc_x', 'Dc_y', 'Dc_z', 'J8', 'J9a', 'J9b']:
        params.add(name, value=0.0)
    load_system(params, show_struct=True, silent=False)
