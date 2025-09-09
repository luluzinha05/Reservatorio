###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from ipywidgets import interact, IntSlider # type: ignore
#import BC_IC_tools as BC_IC
###############################################################################

###############################################################################
class BoundaryCondition:
    def apply_to_pres(self, trans, system_matrix, rhs_vector):
        raise NotImplementedError("Método deve ser implementado pelas subclasses")
    
    def apply_to_vel(self, p, v):
        raise NotImplementedError("Método deve ser implementado pelas subclasses")
###############################################################################

###############################################################################
class DirichletBC1D(BoundaryCondition):
    def __init__(self, PL,PR):
        self.left_pressure_value  = PL  # Valor da pressão no contorno
        self.right_pressure_value = PR  # Valor da pressão no contorno
    
    def apply_to_pres(self, trans, system_matrix, rhs_vector):
        # Left side
        idx = 0          # Índice na matriz/vetor
        idx2 = idx + 1   # Índice adjacente
        
        Ta = trans[idx]  # Transmissibilidade na fronteira
        Tb = trans[idx2]

        system_matrix[idx, idx] = -(Ta+Tb)  # Coeficiente na diagonal
        system_matrix[idx, idx2] = Tb
        rhs_vector[idx] -= Ta * self.left_pressure_value

        # Right side
        idx = -1         # Índice na matriz/vetor
        idx2 = idx - 1   # Índice adjacente
        
        Ta = trans[idx]  # Transmissibilidade na fronteira
        Tb = trans[idx2]

        system_matrix[idx, idx] = -(Ta+Tb)  # Coeficiente na diagonal
        system_matrix[idx, idx2] = Tb
        rhs_vector[idx] -= Ta * self.right_pressure_value

    def apply_to_vel(self, trans, p, v):
        # Left side
        idx = 0
        v[idx] = -trans[idx]*(p[idx] - self.left_pressure_value)  # Transmissibilidade na fronteira   

        # Right side
        idx = -1
        v[idx] = trans[idx]*(p[idx] - self.right_pressure_value)  # Transmissibilidade na fronteira   
###############################################################################

###############################################################################
class DirichletBC2D(BoundaryCondition):
    def __init__(self, PL, PR):
        self.left_pressure_value  = PL  # Valor da pressão no contorno
        self.right_pressure_value = PR  # Valor da pressão no contorno
        
    def apply_to_pres(self, tx, ty, A, b, nx, ny):
        
        # ====== Interior ======
        # Left side
        i = 0
        for j in range(1, ny-1):
            idx = idx_2d_to_1d(i, j, nx)

            tx_right = tx[idx_2d_to_1d(i+1, j, nx+1)]
            tx_left  = tx[idx_2d_to_1d(i,   j, nx+1)]
            ty_up    = ty[idx_2d_to_1d(i, j+1, nx)]
            ty_down  = ty[idx_2d_to_1d(i, j,   nx)]
                
            A[idx, idx] = -(tx_left + tx_right + ty_down + ty_up)
            A[idx, idx_2d_to_1d(i+1, j, nx)] = tx_right
            A[idx, idx_2d_to_1d(i, j+1, nx)] = ty_up
            A[idx, idx_2d_to_1d(i, j-1, nx)] = ty_down
            b[idx] -= tx_left * self.left_pressure_value
        
        # Right side
        i = nx-1
        for j in range(1, ny-1):
            idx = idx_2d_to_1d(i, j, nx)

            tx_right = tx[idx_2d_to_1d(i+1, j, nx+1)]
            tx_left  = tx[idx_2d_to_1d(i,   j, nx+1)]
            ty_up    = ty[idx_2d_to_1d(i, j+1, nx)]
            ty_down  = ty[idx_2d_to_1d(i, j,   nx)]

            A[idx, idx] = -(tx_left + tx_right + ty_down + ty_up)
            A[idx, idx_2d_to_1d(i-1, j, nx)] = tx_left
            A[idx, idx_2d_to_1d(i, j+1, nx)] = ty_up
            A[idx, idx_2d_to_1d(i, j-1, nx)] = ty_down
            b[idx] -= tx_right * self.right_pressure_value

        # Bottom side
        j = 0
        for i in range(1, nx-1):
            idx = idx_2d_to_1d(i, j, nx)

            tx_right = tx[idx_2d_to_1d(i+1, j, nx+1)]
            tx_left  = tx[idx_2d_to_1d(i,   j, nx+1)]
            ty_up    = ty[idx_2d_to_1d(i, j+1, nx)]
            ty_down  = 0.0

            A[idx, idx] = -(tx_left + tx_right + ty_down + ty_up)
            A[idx, idx_2d_to_1d(i-1, j, nx)] = tx_left
            A[idx, idx_2d_to_1d(i+1, j, nx)] = tx_right
            A[idx, idx_2d_to_1d(i, j+1, nx)] = ty_up

        # Top side
        j = ny-1
        for i in range(1, nx-1):
            idx = idx_2d_to_1d(i, j, nx)

            tx_right = tx[idx_2d_to_1d(i+1, j, nx+1)]
            tx_left  = tx[idx_2d_to_1d(i,   j, nx+1)]
            ty_up    = 0.0
            ty_down  = ty[idx_2d_to_1d(i, j-1, nx)]

            A[idx, idx] = -(tx_left + tx_right + ty_down + ty_up)
            A[idx, idx_2d_to_1d(i-1, j, nx)] = tx_left
            A[idx, idx_2d_to_1d(i+1, j, nx)] = tx_right
            A[idx, idx_2d_to_1d(i, j-1, nx)] = ty_down

        # ====== CORNERS ======
        # Left bottom
        i, j = 0, 0
        idx = idx_2d_to_1d(i, j, nx)

        tx_right = tx[idx_2d_to_1d(i+1, j, nx+1)]
        tx_left  = tx[idx_2d_to_1d(i,   j, nx+1)]
        ty_up    = ty[idx_2d_to_1d(i, j+1, nx)]
        ty_down  = 0.0

        A[idx, idx] = -(tx_left + tx_right + ty_down + ty_up)
        A[idx, idx_2d_to_1d(i+1, j, nx)] = tx_right
        A[idx, idx_2d_to_1d(i, j+1, nx)] = ty_up
        b[idx] -= (tx_left + ty_down) * self.left_pressure_value

        # Right bottom
        i, j = nx-1, 0
        idx = idx_2d_to_1d(i, j, nx)

        tx_right = tx[idx_2d_to_1d(i+1, j, nx+1)]
        tx_left  = tx[idx_2d_to_1d(i,   j, nx+1)]
        ty_up    = ty[idx_2d_to_1d(i, j+1, nx)]
        ty_down  = 0.0

        A[idx, idx] = -(tx_left + tx_right + ty_down + ty_up)
        A[idx, idx_2d_to_1d(i-1, j, nx)] = tx_left
        A[idx, idx_2d_to_1d(i, j+1, nx)] = ty_up
        b[idx] -= (tx_right + ty_down) * self.right_pressure_value

        # Left top
        i, j = 0, ny-1
        idx = idx_2d_to_1d(i, j, nx)

        tx_right = tx[idx_2d_to_1d(i+1, j, nx+1)]
        tx_left  = tx[idx_2d_to_1d(i,   j, nx+1)]
        ty_up    = 0.0
        ty_down  = ty[idx_2d_to_1d(i, j-1, nx)]

        A[idx, idx] = -(tx_left + tx_right + ty_down + ty_up)
        A[idx, idx_2d_to_1d(i+1, j, nx)] = tx_right
        A[idx, idx_2d_to_1d(i, j-1, nx)] = ty_down
        b[idx] -= (tx_left + ty_up) * self.left_pressure_value

        # Right top
        i, j = nx-1, ny-1
        idx = idx_2d_to_1d(i, j, nx)

        tx_right = tx[idx_2d_to_1d(i+1, j, nx+1)]
        tx_left  = tx[idx_2d_to_1d(i,   j, nx+1)]
        ty_up    = 0.0
        ty_down  = ty[idx_2d_to_1d(i, j-1, nx)]

        A[idx, idx] = -(tx_left + tx_right + ty_down + ty_up)
        A[idx, idx_2d_to_1d(i-1, j, nx)] = tx_left
        A[idx, idx_2d_to_1d(i, j-1, nx)] = ty_down
        b[idx] -= (tx_right + ty_up) * self.right_pressure_value

    def apply_to_vel(self, tx,ty, p, vx,vy, nx,ny):

        # Left side (x = 0)
        vx[0, :ny] = -tx[0, :ny] * (p[0, :ny] - self.left_pressure_value)

        # Right side (x = nx)
        vx[nx, :ny] = -tx[nx, :ny] * (self.right_pressure_value - p[nx-1, :ny])

        # Bottom side (y = 0), Neumann homogênea
        vy[:nx, 0] = 0.0

        # Top side (y = ny), Neumann homogênea
        vy[:nx, ny] = 0.0
###############################################################################

###############################################################################
class DirichletBC3D(BoundaryCondition):
    def __init__(self, PL, PR):
        self.left_pressure_value  = PL  # Valor da pressão no contorno
        self.right_pressure_value = PR  # Valor da pressão no contorno
        
    def apply_to_pres(self, tx, ty, tz, A, b, nx, ny, nz):
        
        faces = {
            "left": 0,
            "right": nx - 1,
            "bottom": 0,
            "top": nz - 1,
            "front": 0,
            "back": ny - 1
        }

        i_min, i_max = 1, faces['right']  # nx-1
        j_min, j_max = 1, faces['back']  # ny-1
        k_min, k_max = 1, faces['top']   # nz-1

                
        # ====== Interior (face) ======
        #(1) Left face
        i = faces['left']
        for j in range(j_min, j_max):
            for k in range(k_min, k_max):
                idx = idx_3d_to_1d(i, j, k, nx, ny)

                tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
                tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
                ty_back    = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
                ty_front  = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
                tz_up = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
                tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

                A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
                #A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
                A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
                A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
                A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
                A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
                A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
                b[idx] -= tx_left * self.left_pressure_value
            
        #(2) Right face
        i = faces['right']
        for j in range(j_min, j_max):
            for k in range(k_min, k_max):
                idx = idx_3d_to_1d(i, j, k, nx, ny)

                tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
                tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
                ty_up    = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
                ty_down  = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
                tz_front = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
                tz_back  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

                A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
                A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
                #A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
                A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_down
                A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_up
                A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_back
                A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_front
                b[idx] -= tx_right * self.right_pressure_value

        #(3) Front face
        j = faces['front']
        for k in range(k_min, k_max):
            for i in range(i_min, i_max):
                idx = idx_3d_to_1d(i, j, k, nx, ny)

                tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
                tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
                ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
                ty_front = 0.0 #ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
                tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
                tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

                A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
                A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
                A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
                #A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
                A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
                A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
                A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up

        #(4) Back face
        j = faces['back']
        for k in range(k_min, k_max):
            for i in range(i_min, i_max):
                idx = idx_3d_to_1d(i, j, k, nx, ny)

                tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
                tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
                ty_back  = 0.0 #ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
                ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
                tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
                tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

                A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
                A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
                A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
                A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
                #A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
                A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
                A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up

        #(5) Bottom face
        k = faces['bottom']
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                idx = idx_3d_to_1d(i, j, k, nx, ny)

                tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
                tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
                ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
                ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
                tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
                tz_down  = 0.0 #tz[idx_3d_to_1d(i, j, k,   nx, ny)]

                A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
                A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
                A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
                A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
                A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
                #A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
                A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up

        #(6) Front face
        k = faces['top']
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                idx = idx_3d_to_1d(i, j, k, nx, ny)

                tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
                tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
                ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
                ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
                tz_up    = 0.0 #tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
                tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

                A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
                A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
                A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
                A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
                A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
                A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
                #A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up

        # ====== Interior (edges) ======
        #(1) Left front
        i, j = faces['left'], faces['front']
        for k in range(k_min, k_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = 0.0 #ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            #A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            #A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
            b[idx] -= tx_left * self.left_pressure_value

        #(2) Right front
        i, j = faces['right'], faces['front']
        for k in range(k_min, k_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = 0.0 #ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            #A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            #A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
            b[idx] -= tx_right * self.right_pressure_value

        #(3) Left back
        i, j = faces['left'], faces['back']
        for k in range(k_min, k_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = 0.0 #ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            #A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            #A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
            b[idx] -= tx_left * self.left_pressure_value

        #(4) Right back
        i, j = faces['right'], faces['back']
        for k in range(k_min, k_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = 0.0 #ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            #A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            #A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
            b[idx] -= tx_right * self.right_pressure_value

        #(5) Front bottom
        j,k = faces['front'], faces['bottom']
        for i in range(i_min, i_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = 0.0 #ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = 0.0 #tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            #A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            #A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
            
        #(6) Back bottom
        j,k = faces['back'], faces['bottom']
        for i in range(i_min, i_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = 0.0 #ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = 0.0 #tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            #A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            #A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up

        #(7) Front top
        j,k = faces['front'], faces['top']
        for i in range(i_min, i_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = 0.0 #ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = 0.0 #tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            #A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            #A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up

        #(8) Back top
        j,k = faces['back'], faces['top']        
        for i in range(i_min, i_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = 0.0 #ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = 0.0 #tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            #A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            #A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up

        #(9) Left bottom
        i,k = faces['left'], faces['bottom']
        for j in range(j_min, j_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = 0.0 #tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            #A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            #A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
            b[idx] -= tx_left * self.left_pressure_value

        #(10) Right bottom
        i,k = faces['right'], faces['bottom']
        for j in range(j_min, j_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = 0.0 #tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            #A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            #A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
            b[idx] -= tx_right * self.right_pressure_value

        #(11) Left top
        i,k = faces['left'], faces['top']
        for j in range(j_min, j_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = 0.0 #tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            #A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            #A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
            b[idx] -= tx_left * self.left_pressure_value

        #(12) Right top       
        i,k = faces['right'], faces['top']
        for j in range(j_min, j_max):
            idx = idx_3d_to_1d(i, j, k, nx, ny)

            tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
            tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
            ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
            ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
            tz_up    = 0.0 #tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
            tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

            A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
            A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
            #A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
            A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
            A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
            A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
            #A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
            b[idx] -= tx_right * self.right_pressure_value

        # ====== CORNERS ======
        #(1) Left front bottom
        i,j,k = faces['left'], faces['front'], faces['bottom']
        idx = idx_3d_to_1d(i, j, k, nx, ny)

        tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
        tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
        ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
        ty_front = 0.0 #ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
        tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
        tz_down  = 0.0 #tz[idx_3d_to_1d(i, j, k,   nx, ny)]

        A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
        #A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
        A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
        #A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
        A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
        #A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
        A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
        b[idx] -= tx_left * self.left_pressure_value

        #(2) Right front bottom
        i,j,k = faces['right'], faces['front'], faces['bottom']
        idx = idx_3d_to_1d(i, j, k, nx, ny)

        tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
        tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
        ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
        ty_front = 0.0 #ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
        tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
        tz_down  = 0.0 #tz[idx_3d_to_1d(i, j, k,   nx, ny)]

        A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
        A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
        #A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
        #A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
        A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
        #A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
        A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
        b[idx] -= tx_right * self.right_pressure_value

        #(3) Left back bottom
        i,j,k = faces['left'], faces['back'], faces['bottom']
        idx = idx_3d_to_1d(i, j, k, nx, ny)

        tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
        tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
        ty_back  = 0.0 #ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
        ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
        tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
        tz_down  = 0.0 #tz[idx_3d_to_1d(i, j, k,   nx, ny)]

        A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
        #A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
        A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
        A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
        #A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
        #A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
        A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
        b[idx] -= tx_left * self.left_pressure_value

        #(4) Right back bottom
        i,j,k = faces['right'], faces['back'], faces['bottom']
        idx = idx_3d_to_1d(i, j, k, nx, ny)

        tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
        tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
        ty_back  = 0.0 #ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
        ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
        tz_up    = tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
        tz_down  = 0.0 #tz[idx_3d_to_1d(i, j, k,   nx, ny)]

        A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
        A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
        #A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
        A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
        #A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
        #A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
        A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
        b[idx] -= tx_right * self.right_pressure_value

        #(5) Left front top
        i,j,k = faces['left'], faces['front'], faces['top']
        idx = idx_3d_to_1d(i, j, k, nx, ny)

        tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
        tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
        ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
        ty_front = 0.0 #ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
        tz_up    = 0.0 #tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
        tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

        A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
        #A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
        A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
        #A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
        A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
        A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
        #A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
        b[idx] -= tx_left * self.left_pressure_value

        #(6) Right front top
        i,j,k = faces['right'], faces['front'], faces['top']
        idx = idx_3d_to_1d(i, j, k, nx, ny)

        tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
        tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
        ty_back  = ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
        ty_front = 0.0 #ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
        tz_up    = 0.0 #tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
        tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

        A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
        A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
        #A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
        #A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
        A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
        A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
        #A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
        b[idx] -= tx_right * self.right_pressure_value

        #(7) Left back top
        i,j,k = faces['left'], faces['back'], faces['top']
        idx = idx_3d_to_1d(i, j, k, nx, ny)

        tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
        tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
        ty_back  = 0.0 #ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
        ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
        tz_up    = 0.0 #tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
        tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

        A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
        #A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
        A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
        A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
        #A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
        A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
        #A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
        b[idx] -= tx_left * self.left_pressure_value

        #(8) Right back top
        i,j,k = faces['right'], faces['back'], faces['top']
        idx = idx_3d_to_1d(i, j, k, nx, ny)

        tx_right = tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
        tx_left  = tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
        ty_back  = 0.0 #ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
        ty_front = ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
        tz_up    = 0.0 #tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
        tz_down  = tz[idx_3d_to_1d(i, j, k,   nx, ny)]

        A[idx, idx] = -(tx_right + tx_left + ty_front + ty_back + tz_up + tz_down)
        A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
        #A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
        A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_front
        #A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_back
        A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_down
        #A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_up
        b[idx] -= tx_right * self.right_pressure_value

    def apply_to_vel(self, tx,ty,tz, p, vx,vy,vz, nx,ny,nz):

                # Left face (x = 0): Dirichlet
        vx[0, :ny, :nz] = -tx[0, :ny, :nz] * (p[0, :ny, :nz] - self.left_pressure_value)

        # Right face (x = nx): Dirichlet
        vx[nx, :ny, :nz] = -tx[nx, :ny, :nz] * (self.right_pressure_value - p[nx-1, :ny, :nz])

        # Bottom face (y = 0): Neumann homogênea (vy = 0)
        vy[:, 0, :] = 0.0

        # Top face (y = ny): Neumann homogênea (vy = 0)
        vy[:, ny, :] = 0.0

        # Front face (z = 0): Neumann homogênea (vz = 0)
        vz[:, :, 0] = 0.0

        # Back face (z = nz): Neumann homogênea (vz = 0)
        vz[:, :, nz] = 0.0
###############################################################################

###############################################################################
class NeumannBC1D(BoundaryCondition):
    def __init__(self, flux_value):
        self.flux_value = flux_value  # Valor do fluxo no contorno
    
    def apply_to_pres(self, trans, system_matrix, rhs_vector, position):
        idx = 0 if position == 'left' else -1
        sign = 1 if position == 'left' else -1  # Convenção de sinal
        
        # Apenas modifica o vetor RHS
        rhs_vector[idx] += sign *self.flux_value/()

    def apply_to_vel(self, trans, p, v, position):
        idx = 0 if position == 'left' else -1  # Índice na matriz/vetor
        v[idx] = self.flux_value
###############################################################################

###############################################################################
class InitialCondition:
    def create_apply(self, nx, ny, nz):
        raise NotImplementedError("Método deve ser implementado pelas subclasses")
    
    def mask(self, X, Y, Z):
        raise NotImplementedError("Método deve ser implementado pelas subclasses")

    def get_analytic_solution(self, X, Y, Z, t, ux, uy, uz):
        X_shift = X - ux * t
        Y_shift = Y - uy * t
        Z_shift = Z - uz * t

        oil_mask = self.mask(X_shift, Y_shift, Z_shift)
        return (oil_mask).astype(float)
###############################################################################

###############################################################################
class InitialConditionFullOil(InitialCondition):
    """Meio completamente preenchido com óleo (sem traçador)."""
    
    def create_apply(self, nx, ny, nz):
        return np.zeros((nx+2, ny+2, nz+2))

    def mask(self, X, Y, Z):
        return (X < 0)
###############################################################################

###############################################################################
class InitialConditionOilStain(InitialCondition):
    """Mancha de óleo (sem traçador)."""
    def __init__(self, x0, x1, y0, y1, z0, z1, dx, dy, dz):
        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = y0, y1
        self.z0, self.z1 = z0, z1
        self.dx = dx
        self.dy = dy
        self.dz = dz
    
    def create_apply(self, nx, ny, nz):
        c = np.ones((nx+2, ny+2, nz+2))

        ix0 = int(self.x0 / self.dx) + 1
        ix1 = int(self.x1 / self.dx) + 1
        iy0 = int(self.y0 / self.dy) + 1
        iy1 = int(self.y1 / self.dy) + 1
        iz0 = int(self.z0 / self.dz) + 1
        iz1 = int(self.z1 / self.dz) + 1

        c[ix0:ix1, iy0:iy1, iz0:iz1] = 0.0
        return c

    def mask(self, X, Y, Z):
        return ~((self.x0 <= X) & (X <= self.x1) 
                 & (self.y0 <= Y) & (Y <= self.y1) 
                 #& (self.z0 <= Z) & (Z <= self.z1)
                 )
###############################################################################

###############################################################################
class simulationpar:
    '''Description: This class contains the simulation parameters'''
    def __init__(self, simul_setup, beta, rho, mu, Dom, mesh, BHP, PR, PL,
                 rw, q, pos, numb_par):
        self.simul_setup = simul_setup
        self.beta = beta
        self.rho  = rho
        self.mu   = mu
        self.Dom  = Dom
        self.mesh = mesh
        self.BHP  = BHP
        self.PR   = PR.item()
        self.PL   = PL.item()
        self.rw   = rw
        self.q    = q
        self.positions = pos
        self.numb_Gauss_par = numb_par
###############################################################################

###############################################################################
def search_str_by_line(file_path, word, functype):
    '''Description: This function searches for a string in a file and returns'''
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        # read all content of a file
        content = line.strip()
        # check if string present in a file
        if word in content:
            strnum = content[content.find(':')+1:len(content)]
            strnum = strnum.strip()
            if strnum.isdigit() == True:
                strnum = functype(strnum)
    return strnum
###############################################################################

###############################################################################
def fivespot2Dconf(filename):
    '''Description: This function reads the five-spot configuration'''
    BHP = np.fromstring(search_str_by_line(filename,
                                           'Bottom hole pressure (BHP):',
                                           np.int64), dtype = float, sep = ' ')
    rw = np.fromstring(search_str_by_line(filename,
                                           'Well radius (rw):',
                                           np.int64), dtype = float, sep = ' ')
    q  = np.fromstring(search_str_by_line(filename,
                                           'Production rate (q):',
                                           np.int64), dtype = float, sep = ' ')
    return BHP, rw, q
###############################################################################

###############################################################################
def slab2Dconf(filename):
    '''Description: This function reads the slab configuration'''
    PR = np.fromstring(search_str_by_line(filename,
                                          'Right side Dirichlet pressure:',
                                           np.int64), dtype = float, sep = ' ')
    PL = np.fromstring(search_str_by_line(filename,
                                          'Left side Dirichlet pressure:',
                                           np.int64), dtype = float, sep = ' ')
    return PR, PL
###############################################################################

###############################################################################
def input_simulation_parameters(filename):
    '''Description: This function reads the input simulation parameters'''
    simul_setup = ''
    Dom  = [0., 0., 0.]
    mesh = [1, 1, 1]
    BHP, PR, PL, rw, q, mu, beta, rho = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    pos, numb_Gauss_par = 0, 0
    # =========================================================================
    if filename != None:
        simul_setup = search_str_by_line(filename,
                                         'Simulation domain configuration:',
                                         np.int64)
        numb_Gauss_par = search_str_by_line(filename,
                                         'Number of Gaussian parameter:',
                                         np.int64)
        numb_Gauss_par = numb_Gauss_par - 1
        beta = np.fromstring(
            search_str_by_line(filename,'Beta parameter of the permeability field:',
                               np.int64), dtype = float, sep = ' ')
        rho = np.fromstring(
            search_str_by_line(filename,
                               'Rho (strength) parameter of the permeability field:',
                               np.int64), dtype = float, sep = ' ')
        mu = np.fromstring(search_str_by_line(filename, 'Water viscosity (mu):',
                                              np.int64), dtype = float, sep = ' ')
        L  = search_str_by_line(filename, 'Domain size:', np.int64)
        Dom= np.zeros((3,), dtype = 'float')
        Dom= np.fromstring(L, dtype = float, sep = ' ')
        M  = search_str_by_line(filename, 'Computational mesh:', np.int64)
        mesh= np.zeros((3,), dtype = 'int')
        mesh= np.fromstring(M, dtype = int, sep = ' ')
        pos = np.fromstring(search_str_by_line(filename, 'Monitor cell positions:',
                                               np.int64), dtype = int, sep = ',')
        # =====================================================================
        if simul_setup == 'slab2D':
            PR, PL = slab2Dconf(filename)
            q = None
        # =====================================================================
        if simul_setup == 'fivespot2D':
            BHP, rw, q = fivespot2Dconf(filename)
        # =====================================================================
    return simulationpar(simul_setup,beta,rho,mu,Dom,mesh,BHP,PR,PL,
                         rw,q,pos,numb_Gauss_par)
###############################################################################

##############################################################################
def coordinates1D(nx, Lx):
    '''Generate the grid coordinates'''
    dx = Lx / nx
    x = np.linspace(dx/2, Lx-dx/2, nx)
    coord = np.zeros((nx, 1))
    idx   = np.zeros((nx, 1), dtype = 'int')
    for i in range(nx):
        coord[i] = x[i]
        idx[i] = i
    return idx, coord
###############################################################################

###############################################################################
def coordinates2D(nx, ny, Lx, Ly):
    '''Generate the grid coordinates'''
    dx = Lx / nx
    dy = Ly / ny

    x = np.linspace(dx/2, Lx-dx/2, nx)
    y = np.linspace(dy/2, Ly-dy/2, ny)
    
    coord = np.zeros((nx*ny, 2))
    idx   = np.zeros((nx*ny, 2), dtype = 'int')
    
    for j in range(ny):
        for i in range(nx):
            idx1D = idx_2d_to_1d(i,j,nx)
            coord[idx1D, 0] = x[i]
            coord[idx1D, 1] = y[j]
            idx[idx1D, 0] = i
            idx[idx1D, 1] = j
    return idx, coord
###############################################################################

###############################################################################
def coordinates3D(nx, ny, nz, Lx, Ly, Lz):
    '''Generate the grid coordinates'''
    dx = GLx / Gnx
    dy = GLy / Gny
    dz = GLz / Gnz
    
    x = np.linspace(dx/2, Lx-dx/2, nx)
    y = np.linspace(dy/2, Ly-dy/2, ny)
    z = np.linspace(dz/2, Lz-dz/2, nz)
    
    coord = np.zeros((nx*ny*nz, 3))
    idx   = np.zeros((nx*ny*nz, 3), dtype = 'int')
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx1D = idx_3d_to_1d(i,j,k,nx,ny)
                coord[idx1D, 0] = x[i]
                coord[idx1D, 1] = y[j]
                coord[idx1D, 2] = z[k]
                idx[idx1D, 0] = i
                idx[idx1D, 1] = j
                idx[idx1D, 2] = k
    return idx, coord
###############################################################################

###############################################################################
def plot_pres1D(coord,pos,nx,conduct,pres,data):
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(131)
    ax.set_title('Hydraulic Conductivity Field')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Hydraulic Conductivity')
    #ax.plot(coord,conduct)
    cmap = plt.get_cmap('jet')
    vmin = conduct.min()
    vmax = conduct.max()
    vmin, vmax = (vmin - 0.1*vmin, vmin + 0.1*vmin) if vmin == vmax else (vmin, vmax)
    norm = plt.Normalize(vmin,vmax)
    line_colors = cmap(norm(conduct))
    #fig.colorbar(c, ax=ax)

    ax.scatter(coord,conduct,color=line_colors,s=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # necessário para que a colorbar funcione
    fig.colorbar(sm,ax=ax)
    ax.grid(True)

    ax = fig.add_subplot(132)
    ax.set_title('Pressure Field')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Pressure (Pa)')
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(pres.min(), pres.max())
    line_colors = cmap(norm(pres))

    ax.scatter(coord,pres,color=line_colors,s=8,zorder=5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # necessário para que a colorbar funcione
    fig.colorbar(sm,ax=ax)
    ax.grid(True)
    ax.plot(coord[pos].reshape((len(data),)), data, 'k+', markersize=8,zorder=10)

    ax = fig.add_subplot(133)
    ax.set_title('Pressure in sensors')
    ax.set_xlabel('Positions')
    ax.set_ylabel('Pressure (Pa)')
    ax.plot(np.arange(len(data)), data, 'o', label='Pres Flat')
    ax.grid(True)

    plt.tight_layout()
    plt.show()
###############################################################################

###############################################################################
def plot_pres2D(coord,pos,nx,ny,conduct,pres,data):
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(131)
    ax.set_title('Hydraulic Conductivity Field')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    c = ax.pcolormesh(coord[:,0].reshape((ny,nx), order='C'), 
                      coord[:,1].reshape((ny,nx), order='C'), 
                      conduct.T, shading='auto', cmap='jet')
    fig.colorbar(c, ax=ax)
    #ax.plot(coord,conduct)
    cmap = plt.get_cmap('jet')
    #vmin = conduct.min()
    #vmax = conduct.max()
    #vmin, vmax = (vmin - 0.1*vmin, vmin + 0.1*vmin) if vmin == vmax else (vmin, vmax)
    #norm = plt.Normalize(vmin,vmax)
    #line_colors = cmap(norm(conduct))
    #fig.colorbar(c, ax=ax)

    #ax.scatter(coord,conduct,color=line_colors,s=8)
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])  # necessário para que a colorbar funcione
    #fig.colorbar(sm,ax=ax)
    #ax.grid(True)

    ax = fig.add_subplot(132)
    #ax.set_title('Pressure Field')
    #ax.set_xlabel('X Coordinate')
    #ax.set_ylabel('Pressure (Pa)')
    #cmap = plt.get_cmap('jet')
    #norm = plt.Normalize(pres.min(), pres.max())
    #line_colors = cmap(norm(pres))

    #ax.scatter(coord,pres,color=line_colors,s=8,zorder=5)
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])  # necessário para que a colorbar funcione
    #fig.colorbar(sm,ax=ax)
    #ax.grid(True)
    #ax.plot(coord[pos].reshape((len(data),)), data, 'k+', markersize=8,zorder=10)

    ax.set_title('Pressure Field')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    c = ax.pcolormesh(coord[:,0].reshape((ny,nx), order='C'), 
                    coord[:,1].reshape((ny,nx), order='C'), 
                    pres.T, shading='auto', cmap='jet')
    fig.colorbar(c, ax=ax)
    ax.plot(coord[pos,0], coord[pos,1], 'k+', markersize=8)

    ax = fig.add_subplot(133)
    ax.set_title('Pressure in sensors')
    ax.set_xlabel('Positions')
    ax.set_ylabel('Pressure (Pa)')
    ax.plot(np.arange(len(data)), data, 'o', label='Pres Flat')
    ax.grid(True)

    plt.tight_layout()
    plt.show()
###############################################################################

###############################################################################
def compute_perm(Y, rho, beta):
    '''Generate a heterogeneous permeability field'''
    return  beta * np.exp(rho * Y)
###############################################################################

###############################################################################
def compute_hmean(a, b):
    denom = a + b
    hmean = np.where(denom == 0, 0.0, 2 * a * b / denom)
    return hmean
###############################################################################

###############################################################################
def compute_trans_1D(perm, dx, area, viscosidade):

    N = len(perm)
    trans = np.zeros(N+1)

    # Fator comum para todas as transmissibilidades internas
    fator = area / (viscosidade * dx)
    
    # Transmissibilidades internas (faces 1 a N-1)
    trans[1:N] = compute_hmean(perm[:-1], perm[1:]) * fator
    
    # Condições de contorno (faces 0 e N)
    # Fator diferente para contornos (dx/2)
    fator_contorno = 2*area / (viscosidade * dx)
    trans[0] = perm[0] * fator_contorno    # Face esquerda
    trans[-1] = perm[-1] * fator_contorno  # Face direita
    
    return trans
###############################################################################

###############################################################################
def compute_trans_2D(perm, dx, dy, profundidade, viscosidade):

    nx, ny = perm.shape
    Tx = np.zeros((nx+1, ny))  # transmissibilidade faces verticais (entre colunas)
    Ty = np.zeros((nx, ny+1))  # transmissibilidade faces horizontais (entre linhas)

    # Fator transmissibilidade para faces verticais (x)
    area_x = dy * profundidade
    fator_x = area_x / (viscosidade * dx)

    # Fator transmissibilidade para faces horizontais (y)
    area_y = dx * profundidade
    fator_y = area_y / (viscosidade * dy)

    # Transmissibilidades internas x (entre colunas)
    perm_x1 = perm[:-1, :]  # células à esquerda
    perm_x2 = perm[1:, :]   # células à direita
    Tx[1:-1, :] = compute_hmean(perm_x1, perm_x2) * fator_x  # por causa do dx médio

    # Transmissibilidades internas y (entre linhas)
    perm_y1 = perm[:, :-1]  # células abaixo
    perm_y2 = perm[:, 1:]   # células acima
    Ty[:, 1:-1] = compute_hmean(perm_y1, perm_y2) * fator_y  # por causa do dy médio

    # Condições de contorno
    Tx[0, :]  = 2 * perm[0, :] * area_x / (viscosidade * dx)  # Contorno esquerda (face 0)
    Tx[-1, :] = 2 * perm[-1, :] * area_x / (viscosidade * dx) # Contorno direita (face nx)
    Ty[:, 0]  = 2 * perm[:, 0] * area_y / (viscosidade * dy)  # Contorno inferior (face 0)
    Ty[:, -1] = 2 * perm[:, -1] * area_y / (viscosidade * dy) # Contorno superior (face ny)

    return Tx, Ty  # Retorna como vetores 2D
###############################################################################

###############################################################################
def compute_trans_3D(perm, dx, dy, dz, viscosidade):

    nx, ny, nz = perm.shape
    
    Tx = np.zeros((nx+1, ny, nz))
    Ty = np.zeros((nx, ny+1, nz))
    Tz = np.zeros((nx, ny, nz+1))
    
    area_x = dy * dz
    area_y = dx * dz
    area_z = dx * dy

    fator_x = area_x / (viscosidade * dx)
    fator_y = area_y / (viscosidade * dy)
    fator_z = area_z / (viscosidade * dz)

    # Transmissibilidades internas - eixo x
    Tx[1:-1, :, :] = compute_hmean(perm[:-1, :, :], perm[1:, :, :]) * fator_x

    # Transmissibilidades internas - eixo y
    Ty[:, 1:-1, :] = compute_hmean(perm[:, :-1, :], perm[:, 1:, :]) * fator_y

    # Transmissibilidades internas - eixo z
    Tz[:, :, 1:-1] = compute_hmean(perm[:, :, :-1], perm[:, :, 1:]) * fator_z

    # Contornos (usando cond. Neumann homogênea implícita = permeabilidade da própria célula)
    Tx[0, :, :]  = 2 * perm[0, :, :] * fator_x
    Tx[-1, :, :] = 2 * perm[-1, :, :] * fator_x

    Ty[:, 0, :]  = 2 * perm[:, 0, :] * fator_y
    Ty[:, -1, :] = 2 * perm[:, -1, :] * fator_y

    Tz[:, :, 0]  = 2 * perm[:, :, 0] * fator_z
    Tz[:, :, -1] = 2 * perm[:, :, -1] * fator_z

    return Tx, Ty, Tz  # Retorna como vetores 3D
###############################################################################

###############################################################################
def idx_2d_to_1d(i, j, nx):
    return j * nx + i
###############################################################################

###############################################################################
def idx_1d_to_2d(idx, nx):
    j = idx // nx
    i = idx % nx
    return i, j
###############################################################################

###############################################################################
def idx_3d_to_1d(i, j, k, nx, ny):
    """
    Converte índice 3D (i, j, k) para índice 1D.

    Parâmetros:
    - i: índice na direção x (0 <= i < nx)
    - j: índice na direção y (0 <= j < ny)
    - k: índice na direção z (0 <= k < nz)
    - nx, ny: tamanho das dimensões x e y

    Retorna:
    - índice 1D correspondente
    """
    return k * (nx * ny) + j * nx + i
###############################################################################

###############################################################################
def idx_1d_to_3d(idx, nx, ny):
    """
    Converte índice 1D para índice 3D (i, j, k).

    Parâmetros:
    - idx: índice 1D
    - nx, ny: tamanho das dimensões x e y

    Retorna:
    - tupla (i, j, k) correspondente
    """
    k = idx // (nx * ny)
    remainder = idx % (nx * ny)
    j = remainder // nx
    i = remainder % nx
    return i, j, k
###############################################################################

###############################################################################
def setup_system_1D(trans, BC, q=None):
    N = len(trans) - 1  # N células = N+1 faces
    A = np.zeros((N, N))  # Matriz do sistema
    b = np.zeros(N) if q is None else -q.copy()  # Vetor RHS
    
    # Montagem das equações internas
        # índices internos (1 a N-2)
    i = np.arange(1, N-1)
    A[i, i] = -(trans[i] + trans[i+1])  # diagonal principal
    A[i, i-1] = trans[i]                # termo da esquerda
    A[i, i+1] = trans[i+1]              # termo da direita

    # Aplicação das condições de contorno
    BC.apply_to_pres(trans, A, b)

    return A, b
###############################################################################

###############################################################################
def setup_system_2D(Tx_, Ty_,BC, q=None):
        
    nx = Tx_.shape[0] - 1
    ny = Ty_.shape[1] - 1
    N = nx * ny
    Tx = Tx_.flatten(order='F')
    Ty = Ty_.flatten(order='F')

    A = lil_matrix((N, N))  # Matriz do sistema
    b = np.zeros(N) if q is None else -q.copy()  # Vetor RHS
    
    # Montagem das equações internas
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            idx = idx_2d_to_1d(i, j, nx)
            
            tx_right = Tx[idx_2d_to_1d(i + 1, j, nx + 1)]
            tx_left  = Tx[idx_2d_to_1d(i,     j, nx + 1)]
            ty_up    = Ty[idx_2d_to_1d(i, j + 1, nx)]
            ty_down  = Ty[idx_2d_to_1d(i, j,     nx)]

            A[idx, idx] = -(tx_right + tx_left + ty_up + ty_down)
            A[idx, idx_2d_to_1d(i-1, j, nx)] = tx_left
            A[idx, idx_2d_to_1d(i+1, j, nx)] = tx_right
            A[idx, idx_2d_to_1d(i, j-1, nx)] = ty_down
            A[idx, idx_2d_to_1d(i, j+1, nx)] = ty_up

    # Aplicação das condições de contorno
    BC.apply_to_pres(Tx, Ty, A, b, nx, ny)
    
    return A.tocsr(), b
###############################################################################

###############################################################################
def setup_system_3D(Tx_, Ty_, Tz_, BC, q=None):
    nx = Tx_.shape[0] - 1
    ny = Ty_.shape[1] - 1
    nz = Tz_.shape[2] - 1
    N = nx * ny * nz

    Tx = Tx_.flatten(order='F')
    Ty = Ty_.flatten(order='F')
    Tz = Tz_.flatten(order='F')

    A = lil_matrix((N, N))  # Matriz do sistema
    b = np.zeros(N) if q is None else -q.copy()  # Vetor RHS
    
    # Montagem das equações internas
    for k in range(1, nz-1):
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                idx = idx_3d_to_1d(i, j, k, nx, ny)

                tx_right = Tx[idx_3d_to_1d(i+1, j, k, nx+1, ny)]
                tx_left  = Tx[idx_3d_to_1d(i,   j, k, nx+1, ny)]
                ty_up    = Ty[idx_3d_to_1d(i, j+1, k, nx, ny+1)]
                ty_down  = Ty[idx_3d_to_1d(i, j,   k, nx, ny+1)]
                tz_front = Tz[idx_3d_to_1d(i, j, k+1, nx, ny)]
                tz_back  = Tz[idx_3d_to_1d(i, j, k,   nx, ny)]

                A[idx, idx] = -(tx_right + tx_left + ty_up + ty_down + tz_front + tz_back)
                A[idx, idx_3d_to_1d(i-1, j, k, nx, ny)] = tx_left
                A[idx, idx_3d_to_1d(i+1, j, k, nx, ny)] = tx_right
                A[idx, idx_3d_to_1d(i, j-1, k, nx, ny)] = ty_down
                A[idx, idx_3d_to_1d(i, j+1, k, nx, ny)] = ty_up
                A[idx, idx_3d_to_1d(i, j, k-1, nx, ny)] = tz_back
                A[idx, idx_3d_to_1d(i, j, k+1, nx, ny)] = tz_front

    # Aplicação das condições de contorno
    BC.apply_to_pres(Tx, Ty, Tz, A, b, nx, ny,nz)
    
    return A.tocsr(), b
###############################################################################

###############################################################################
def compute_tpfa_velocity1D(p, trans,BC):
    N = len(p)
    v = np.zeros(N + 1)
    
    # Velocidades nas faces internas (i=1 a N-1)
    for i in range(1, N):
        delta_p = p[i] - p[i-1]
        v[i] = -trans[i] * delta_p
    
    # Velocidade nas fronteiras
    BC.apply_to_vel(trans,p,v)
    
    return v
###############################################################################

###############################################################################
def compute_tpfa_velocity2D(p,Tx,Ty,BC):
    nx, ny = p.shape
    vx = np.zeros((nx + 1, ny))
    vy = np.zeros((nx, ny + 1))

    # Neste programa considera-se que Tx e Ty já foram divididos por Ax (H*dy) e Ay (H*dx)

    # Velocidade nas faces verticais (x-direction)
    for i in range(1, nx):
        for j in range(ny):
            dp = p[i, j] - p[i - 1, j]
            vx[i, j] = -Tx[i, j] * dp

    # Velocidade nas faces horizontais (y-direction)
    for i in range(nx):
        for j in range(1, ny):
            dp = p[i, j] - p[i, j - 1]
            vy[i, j] = -Ty[i, j] * dp

    # Velocidade nas fronteiras
    BC.apply_to_vel(Tx, Ty, p, vx, vy, nx, ny)
    
    return vx, vy  # Retorna as velocidades nas faces verticais e horizontais
###############################################################################

###############################################################################
def compute_tpfa_velocity3D(p,Tx,Ty,Tz,BC):
    
    nx, ny, nz = p.shape
    vx = np.zeros((nx+1, ny, nz))  # velocidades em x (entre células i-1 e i)
    vy = np.zeros((nx, ny+1, nz))  # velocidades em y (entre j-1 e j)
    vz = np.zeros((nx, ny, nz+1))  # velocidades em z (entre k-1 e k)

    # vx (faces verticais em x)
    for i in range(1, nx):
        for j in range(ny):
            for k in range(nz):
                dp = p[i, j, k] - p[i-1, j, k]
                vx[i, j, k] = -Tx[i, j, k] * dp

    # vy (faces horizontais em y)
    for i in range(nx):
        for j in range(1, ny):
            for k in range(nz):
                dp = p[i, j, k] - p[i, j-1, k]
                vy[i, j, k] = -Ty[i, j, k] * dp

    # vz (faces frontais em z)
    for i in range(nx):
        for j in range(ny):
            for k in range(1, nz):
                dp = p[i, j, k] - p[i, j, k-1]
                vz[i, j, k] = -Tz[i, j, k] * dp

    # Velocidade nas fronteiras
    BC.apply_to_vel(Tx, Ty, Tz, p, vx, vy, vz, nx, ny, nz)
    
    return vx, vy, vz  # Retorna as velocidades nas faces verticais e horizontais
###############################################################################

###############################################################################
def compute_div1D(vx):
    div = (vx[1:] - vx[:-1])

    return div
###############################################################################

###############################################################################
def compute_div2D(vx,vy):
    div_x = (vx[1:, :] - vx[:-1, :])  # shape (nx, ny)
    div_y = (vy[:, 1:] - vy[:, :-1])  # shape (nx, ny)
    div = div_x + div_y

    return div
###############################################################################

###############################################################################
def compute_div3D(vx,vy,vz):
    div_x = (vx[1:, :, :] - vx[:-1, :, :])  # shape (nx, ny, nz)
    div_y = (vy[:, 1:, :] - vy[:, :-1, :])  # shape (nx, ny, nz)
    div_z = (vz[:, :, 1:] - vz[:, :, :-1])  # shape (nx, ny, nz)
    div = div_x + div_y + div_z

    return div
###############################################################################

###############################################################################
def plot_vel_div1D(vx,coord,div,nx,Lx,dx,fact,fact2):

    vmin = np.min(np.min(np.abs(div)),0)
    vmax = np.max(div)

    #x_vx = np.linspace(0,Lx,nx+1)  # nx+1 faces verticais
    #y_vx = 0
    #X_vx, Y_vx = np.meshgrid(x_vx, y_vx)
    #
    x_centers = np.linspace(dx/2, Lx - dx/2, nx)
    y_centers = 0
    X, Y = np.meshgrid(x_centers,y_centers,indexing='ij')

    vx_cell = 0.5 * (vx[:-1] + vx[1:])  # shape (nx, ny)

    fig = plt.figure(figsize=(10*fact, 4*fact))
    ax = fig.add_subplot(121)
    ax.plot([0, Lx], [0, 0], 'black', linestyle='-', linewidth=2,alpha=0.2)
    for i in range(1,nx):
        x = i*dx
        ax.plot([x, x], [-0.01*Lx, 0.01*Lx], 'black', linestyle='-', linewidth=2,alpha=0.2)

    ax.quiver(X, Y, vx_cell.T, np.zeros_like(vx_cell.T), color='blue')
    #ax.quiver(X_vx, Y_vx, vx.T, np.zeros_like(vx.T), color='blue', scale=fact2, label='vx')
    #ax.set_title('Campo de Velocidade nas Faces')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0-0.1*Lx, 0+0.1*Lx)
    #ax.grid(True)
    #plt.legend(["grid zone","vx", "vy"], loc="upper right")

    ax = fig.add_subplot(122)
    ax.set_title('Divergency of velocity')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Divergency')

    cmap = plt.get_cmap('jet')
    vmin = div.min()
    vmax = div.max()
    mint = 10 if vmin == 0 else vmin
    maxt = 10 if vmax == 0 else vmax
    vmin, vmax = (vmin - 0.1*mint, vmax + 0.1*maxt) if vmin == vmax else (vmin, vmax)
    norm = plt.Normalize(vmin,vmax)
    line_colors = cmap(norm(div))
    #fig.colorbar(c, ax=ax)

    ax.scatter(coord,div,color=line_colors,s=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # necessário para que a colorbar funcione
    fig.colorbar(sm,ax=ax)
    ax.grid(True)

    plt.show()
###############################################################################

###############################################################################
def plot_vel_div2D(vx,vy,coord,div,nx,ny,Lx,Ly,dx,dy,fact,fact2):

    vmin = np.min(np.min(np.abs(div)),0)
    vmax = np.max(div)

    #x_vx = np.linspace(0,Lx,nx+1)  # nx+1 faces verticais
    #y_vx = np.linspace(dy/2,Ly-dy/2,ny)  # ny centros verticais
    #X_vx, Y_vx = np.meshgrid(x_vx, y_vx)
    
    #x_vy = np.linspace(dx/2,Lx-dx/2,nx)  # nx+1 faces verticais
    #y_vy = np.linspace(0,Ly,ny+1)  # ny centros verticais
    #X_vy, Y_vy = np.meshgrid(x_vy, y_vy)
    
    x_centers = np.linspace(dx/2, Lx - dx/2, nx)
    y_centers = np.linspace(dy/2, Ly - dy/2, ny)
    X, Y = np.meshgrid(x_centers,y_centers,indexing='ij')
    #X, Y = np.meshgrid(coord[:,0], coord[:,1],indexing='ij')
    
    vx_cell = 0.5 * (vx[:-1, :] + vx[1:, :])  # shape (nx, ny)
    vy_cell = 0.5 * (vy[:, :-1] + vy[:, 1:])  # shape (nx, ny)

    fig = plt.figure(figsize=(10*fact, 4*fact))
    ax = fig.add_subplot(121)
    for i in range(1, nx):
        x = i * dx
        ax.plot([x, x], [0, Ly], 'black', linestyle='--', linewidth=2,alpha=0.2)
    for j in range(1, ny):
        y = j * dy
        ax.plot([0, Lx], [y, y], 'black', linestyle='--', linewidth=2,alpha=0.2)
    #ret = plt.Rectangle((0, 0), Lx, Ly, fill=None, edgecolor='black',linestyle='--', linewidth=2,alpha=0.2)  # Retângulo do reservatório
    #ax.add_patch(ret)

    #ax.quiver(X_vx, Y_vx, vx.T, np.zeros_like(vx.T), color='blue', scale=fact2, label='vx')
    #ax.quiver(X_vy, Y_vy, np.zeros_like(vy.T),vy.T, color='red', scale=fact2, label='vy')
    #ax.scatter(X, Y, s=50/max([nx,ny]), facecolors='blue', edgecolors='blue')
    ax.quiver(X, Y, vx_cell.T, vy_cell.T,color='blue')#, scale=fact2)

    ax.set_title('Campo de Velocidade nas Faces')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    #ax.grid(True)
    #plt.legend(["grid zone","vx", "vy"], loc="upper right")

    ax = fig.add_subplot(122)
    ax.set_title('Divergency of velocity')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    c = ax.pcolormesh(coord[:,0].reshape((ny,nx), order='C'), 
                    coord[:,1].reshape((ny,nx), order='C'), 
                    np.abs(div.T), shading='auto', cmap='jet',vmin=vmin,vmax=vmax)
    fig.colorbar(c, ax=ax)

    plt.show()
###############################################################################

###############################################################################
def plot_level_z(psim, K, Lx, Ly, z_idx,
                 pmin, pmax, kmin, kmax):
    """
    Plota pressão e permeabilidade no nível z = z_idx com escalas fixas globais.

    Parâmetros:
    - psim: pressão (nx, ny, nz)
    - K: permeabilidade (nx, ny, nz)
    - Lx, Ly: tamanhos físicos
    - z_idx: fatia z a mostrar
    - pmin, pmax: limites globais da pressão
    - kmin, kmax: limites globais da permeabilidade
    """
    nx, ny, nz = psim.shape
    assert 0 <= z_idx < nz, "z_idx fora do intervalo"

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))

    im1 = axs[0].imshow(K[:, :, z_idx].T, origin='lower',
                        extent=[0, Lx, 0, Ly], cmap='jet',
                        vmin=kmin, vmax=kmax)
    axs[0].set_title(f'Permeabilidade')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(psim[:, :, z_idx].T, origin='lower',
                        extent=[0, Lx, 0, Ly], cmap='jet',
                        vmin=pmin, vmax=pmax)
    axs[1].set_title(f'Pressão')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.show()
###############################################################################

###############################################################################
def interactive_plot_3D(psim, K, Lx, Ly):
    """
    Cria um widget interativo para navegar nos níveis z do domínio 3D.
    Usa limites globais para os colormaps.
    """
    _, _, nz = psim.shape

    # Valores globais de min/max
    pmin, pmax = np.min(psim), np.max(psim)
    kmin, kmax = np.min(K), np.max(K)

    interact(lambda k: plot_level_z(psim, K, Lx, Ly, k,
                                        pmin, pmax, kmin, kmax),
             k=IntSlider(min=0, max=nz-1, step=1, value=0))
###############################################################################

###############################################################################
def format_seconds_to_hhmmss(seconds, name):
    '''Description: This function formats the seconds to hh:mm:ss'''
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('\n=====================================================')
    print('=====================================================')
    print(name)
    print("Total time elapsed (h:m:s)..............: %02d:%02d:%2.2f" %
          (hours, minutes, seconds))
    print('=====================================================')
    print('=====================================================')
    return hours, minutes, seconds
###############################################################################

###############################################################################
def initialize_domain_3D(Lx, Ly, Lz, nx, ny, nz):
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    x = np.linspace(dx/2, Lx - dx/2, nx)
    y = np.linspace(dy/2, Ly - dy/2, ny)
    z = np.linspace(dz/2, Lz - dz/2, nz)
    return np.meshgrid(x, y, z, indexing='ij'), dx, dy, dz
###############################################################################

###############################################################################
def initialize_velocity_3D(nx, ny, nz, value=(1.0, 0.0, 0.0)):
    # velocidades nas faces
    ux = np.ones((nx+1, ny, nz)) * value[0] # ux nas faces entre células em x: (nx+1, ny, nz)
    uy = np.ones((nx, ny+1, nz)) * value[1] # uy nas faces em y: (nx, ny+1, nz)
    uz = np.ones((nx, ny, nz+1)) * value[2] # uz nas faces em z: (nx, ny, nz+1)
    return ux, uy, uz
###############################################################################

###############################################################################
def initial_condition_3D(nx, ny, nz, type='left_strip'):
    c = np.zeros((nx+2, ny+2, nz+2))
    if type == 'corner_injection':
        c[0, 0, 0] = 0.0
    elif type == 'left_strip':
        c[0, :, :] = 0.0
    elif type == 'block':
        c[:int(nx*0.2), :int(ny*0.2), :int(nz*0.2)] = 1.0
    return c
###############################################################################

###############################################################################
def apply_ghost_cells_3D(c, left_value=1.0, right_value=0.0):
    # x faces (Dirichlet)
    c[0, :, :]  = left_value   # Esquerda (x=0)
    c[-1, :, :] = right_value  # Direita (x=Lx)
    # y faces (Neumann homogênea)
    c[:, 0, :]  = c[:, 1, :]
    c[:, -1, :] = c[:, -2, :]
    # z faces (Neumann homogênea)
    c[:, :, 0]  = c[:, :, 1]
    c[:, :, -1] = c[:, :, -2]
    
    return c
###############################################################################

###############################################################################
def analytic_solution_3D(X, Y, Z, t, 
                         Lx=1.0, Ly=1.0, Lz = 1.0, 
                         ux_val=1.0, uy_val=0.0, uz_val=0.0,
                         mode='simple'):
    """
    Retorna a solução analítica 2D do problema de advecção pura.
    Assumimos velocidade constante (ux_val, uy_val) e condição inicial do tipo bloco no canto.
    """
    if mode == 'simple':

        return (X < ux_val * t).astype(float)
        #return ((X_shift < ux_val * t) & (Y_shift < uy_val * t)).astype(float)
    
    raise NotImplementedError("Modo analítico desconhecido.")
###############################################################################

###############################################################################
def upwind_step_3Dvec(c, ux, uy, uz, dt, dx, dy, dz):
    nx, ny, nz = c.shape[0] - 2, c.shape[1] - 2, c.shape[2] - 2
    c_new = c.copy()

    # Região interna
    cin = c[1:-1, 1:-1, 1:-1]

    # X-direção
    fx_in  = np.where(ux[:-1, :, :] > 0, ux[:-1, :, :] * c[0:-2, 1:-1, 1:-1], ux[:-1, :, :] * cin)
    fx_out = np.where(ux[1:, :, :]  > 0, ux[1:, :, :]  * cin,              ux[1:, :, :]  * c[2:, 1:-1, 1:-1])

    # Y-direção
    fy_in  = np.where(uy[:, :-1, :] > 0, uy[:, :-1, :] * c[1:-1, 0:-2, 1:-1], uy[:, :-1, :] * cin)
    fy_out = np.where(uy[:, 1:, :]  > 0, uy[:, 1:, :]  * cin,               uy[:, 1:, :]  * c[1:-1, 2:, 1:-1])

    # Z-direção
    fz_in  = np.where(uz[:, :, :-1] > 0, uz[:, :, :-1] * c[1:-1, 1:-1, 0:-2], uz[:, :, :-1] * cin)
    fz_out = np.where(uz[:, :, 1:]  > 0, uz[:, :, 1:]  * cin,               uz[:, :, 1:]  * c[1:-1, 1:-1, 2:])

    # Atualização
    c_new[1:-1, 1:-1, 1:-1] -= (
        dt / dx * (fx_out - fx_in)
      + dt / dy * (fy_out - fy_in)
      + dt / dz * (fz_out - fz_in)
    )    

    #c_new[0, :] = 1.0  # manter injeção contínua na lateral esquerda
    return c_new
###############################################################################

###############################################################################
def run_advection_solver_3D(Lx, Ly, Lz, nx, ny, nz, ux, uy, uz, cfl, tf, IC):
    (x, y, z), dx, dy, dz = initialize_domain_3D(Lx, Ly, Lz, nx, ny, nz)
    
    vel_max = max(np.max(np.abs(ux)), np.max(np.abs(uy)), np.max(np.abs(uz)))
    dt = cfl * min(dx, dy, dz) / max(vel_max, 1e-20)
    nt = int(np.ceil(tf / dt))
    dt = tf / nt

    c = IC.create_apply(nx,ny,nz)
    c_hist = [c[1:-1, 1:-1, 1:-1].copy()]

    for _ in range(nt):
        apply_ghost_cells_3D(c,1.0,0.0)
        c = upwind_step_3Dvec(c, ux, uy, uz, dt, dx, dy, dz)
        c_hist.append(c[1:-1, 1:-1, 1:-1].copy())
        
    return x, y, z, c_hist, dt, nt
###############################################################################

###############################################################################
def plot_solution_3D_slice(X, Y, K, c_hist, dt, n, z_idx,
                           Lx, Ly, Lz, ux_val, uy_val, uz_val,
                           cmin,cmax,kmin,kmax,IC):
    """
    Plota a solução numérica e analítica para o instante n*dt na fatia z = z_idx.
    """
    day = 86400  # segundos em um dia
    t = n * dt
    c_num = c_hist[n][:, :, z_idx]
    
    # Solução analítica 3D
    nx, ny = c_num.shape
    c_ana_3D = IC.create_apply(nx,ny,10)
    c_ana_3D = IC.get_analytic_solution(X, Y, Y, t, ux_val, uy_val, uz_val)

    #c_ana_3D = analytic_solution_3D(X, Y, np.zeros_like(X)+z_idx*(Lz/X.shape[2]), t,
    #                                Lx=Lx, Ly=Ly, Lz=Lz, ux_val=ux_val, uy_val=uy_val, uz_val=uz_val)
    c_ana = c_ana_3D[:, :, z_idx]  # fatia z_idx

    norm_c = np.linalg.norm(c_ana)
    norm_c = norm_c if norm_c !=0 else 1e-12
    erro = np.linalg.norm(c_num - c_ana)/norm_c
    print(f't = {t/day:.2f} days, z = {z_idx}, RE = {erro:.2e}')

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axs[0].imshow(K[:, :, z_idx].T, origin='lower',
                        extent=[0, Lx, 0, Ly], cmap='jet',
                        vmin=kmin, vmax=kmax)
    axs[0].set_title(f'Permeabilidade')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(c_num.T, origin='lower',
                        extent=[0, Lx, 0, Ly], cmap='jet',
                        vmin=cmin, vmax=cmax)
    axs[1].set_title(f'Numérica')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(c_ana.T, origin='lower',
                        extent=[0, Lx, 0, Ly], cmap='jet',
                        vmin=cmin, vmax=cmax)
    axs[2].set_title('Analítica')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    fig.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()
###############################################################################

###############################################################################
def interactive_solution_3D(c_hist, K, dt, Lx, Ly, Lz, ux_val, uy_val, uz_val,IC):
    nx, ny, nz = c_hist[0].shape
    (X, Y, Z), _, _, _ = initialize_domain_3D(Lx, Ly, Lz, nx, ny, nz)
    nt = len(c_hist)

    cmin, cmax = np.min(c_hist), np.max(c_hist)
    kmin, kmax = np.min(K), np.max(K)

    interact(lambda n, z_idx: plot_solution_3D_slice(X, Y, K, c_hist, dt, n, z_idx,
                                                     Lx, Ly, Lz, ux_val, uy_val, uz_val,
                                                     cmin,cmax,kmin,kmax,IC),
             n=IntSlider(min=0, max=nt-1, step=1, value=0, description='Tempo'),
             z_idx=IntSlider(min=0, max=nz-1, step=1, value=0, description='z'))
###############################################################################