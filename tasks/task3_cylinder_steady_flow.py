import os
import matplotlib
import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from tensorboardX import SummaryWriter

from net import Net
from utills import visualise


# 定义问题相关参数
area = [0, 1.1, 0, 0.41] # xmin, xmax, ymin, ymax
r = 0.05 # 圆柱半径
pos_x, pos_y = 0.15 + r, 0.15 + r
rho, miu = 1.0, 0.02
Umax, H = 1.0, area[3] - area[2]

# 定义训练相关参数
num = 7
lr = 0.0005
n_inside = 200
n_wall = 1000
max_epochs = 160000
beta = 2.0
load_model = False
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')


class PINN(Net):
    def __init__(self, seq_net = [3, 64, 64, 64, 64, 3], name='MLP', activation=torch.nn.Tanh()):
        super().__init__(seq_net, name, activation)

    def d(self, f, x):
        """
        自动微分函数
        
        :param f: 因变量
        :param x: 自变量
        """
        return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

    def pde_residual(self, output:list, input:list):
        """
        pde 方程残差函数
        
        :param output: [u, v, p]
        :param input: [x, y]
        :param n_train: 单个维度采样点个数
        """
        u = output[:, 0]
        v = output[:, 1]
        p = output[:, 2]

        d = self.d

        dudx = d(u, input)[:, 0]
        dudy = d(u, input)[:, 1]
        dvdx = d(v, input)[:, 0]
        dvdy = d(v, input)[:, 1]

        d2udx2 = d(dudx, input)[:, 0]
        d2udy2 = d(dudy, input)[:, 1]
        d2vdx2 = d(dvdx, input)[:, 0]
        d2vdy2 = d(dvdy, input)[:, 1]

        dpdx = d(p, input)[:, 0]
        dpdy = d(p, input)[:, 1]

        eq1 = dudx + dvdy
        eq2 = u * dudx + v * dudy + (1 / rho) * dpdx - (miu / rho) * (d2udx2 + d2udy2)
        eq3 = u * dvdx + v * dvdy + (1 / rho) * dpdy - (miu / rho) * (d2vdx2 + d2vdy2)
        eqs =  torch.cat((eq1, eq2, eq3), dim=-1)

        return eqs

    def boundary_residual(self, output, input, type = "wall", index = None):
        """
        边界条件残差函数
        
        :param output: [u, v, p]
        :param input: [x, y]
        :param n_train: 单个维度采样点个数
        :param type: 壁面 入口 出口
        """
        u = output[index][:, 0]
        v = output[index][:, 1]
        p = output[index][:, 2]
        y = input[index][:, 1]

        if type == "wall":
            eq1 = u
            eq2 = v
            eqs = torch.cat((eq1, eq2), dim=-1)

        if type == "inlet":
            eq1 = u - 4 * Umax * (H - y) * y / (H**2)
            eq2 = v
            eqs = torch.cat((eq1, eq2), dim=-1)

        if type == "outlet":
            eq1 = p
            eqs = eq1

        return eqs
    
    def read_data(self, ):
        """
        读划分点和 ground true 流场函数
        """

        # 读入划分点数据
        points_data = sio.loadmat('data/steady_data.mat') 
        points_inlet, points_outlet, points_wall = points_data['INLET'][..., :2], points_data['OUTLET'], points_data['WALL'] # 数据格式 xy_c = xy_inside + inlet + outlet + wall
        not_inside_num = points_inlet.shape[0] + points_outlet.shape[0] + points_wall.shape[0]
        points_inside = points_data['XY_c'][:-not_inside_num]

        # 读入仿真流场数据
        ground_true = sio.loadmat('data/steady_Fluent.mat')
        ground_true = np.squeeze(ground_true['field']).T[..., (0, 1, 4, 2, 3)]

        # 打乱顺序
        data = points_inside, points_inlet, points_outlet, points_wall, ground_true
        data = list(map(np.random.permutation, data))
        points_inside = np.concatenate(data[:-1], 0)
        ground_true = data[-1]

        # 边界点下标
        index_boundaries = []
        nodes = points_inside[:, :2]
        num_nodes = nodes.shape[0]
        index_array = np.arange(num_nodes)

        index_boundaries.append(index_array[nodes[:, 0] == area[0]])  # inlet
        index_boundaries.append(index_array[nodes[:, 0] == area[1]])  # outlet
        index_boundaries.append(index_array[nodes[:, 1] == area[3]])  # top
        index_boundaries.append(index_array[nodes[:, 1] == area[2]])  # bottom
        index_boundaries.append(index_array[np.abs((nodes[:, 0]-0.2)**2 + (nodes[:, 1]-0.2)**2 - (r)**2) < 1e-7])  # cylinder

        if nodes.shape[-1] == 3:
            index_boundaries.append(index_array[nodes[:, 2] == 0])  # initial

        return points_inside, ground_true, index_boundaries


def train():
    name = "task3_" + str(num)
    path = "save/" + name + "/"
    isCreated = os.path.exists(path)
    if not isCreated:
        os.makedirs(path)
    writer = SummaryWriter(log_dir = path + 'log')
    
    pinn = PINN([2, 40, 40, 40, 40, 40, 40, 40, 40, 3]).to(device) # x, y -> u, v, p
    if load_model:
        pinn.load_state_dict(torch.load(path + "model_" + str(num) + ".pth"))

    optimizer = torch.optim.Adam(pinn.parameters(), lr, betas=(0.9, 0.999))
    criterion = torch.nn.MSELoss()

    # 读入数据
    points_inside, ground_true, index_boundaries = pinn.read_data()
    input = torch.tensor(points_inside[:, :2], dtype=torch.float32).to(device)
    ground_true = torch.tensor(ground_true, dtype=torch.float32)
    triang = matplotlib.tri.Triangulation(ground_true[:, 0], ground_true[:, 1])
    triang.set_mask(np.hypot(ground_true[triang.triangles, 0].mean(axis=1) - 0.2,
                             ground_true[triang.triangles, 1].mean(axis=1) - 0.2) < r)
    
    vis = visualise()

    # train
    epoch = 0
    loss_log = []
    while epoch < max_epochs:
        epoch += 1
        optimizer.zero_grad()

        input = input.cuda()
        input.requires_grad_(True)
        output = pinn(input)

        # pde loss
        pde_loss = 0
        pde_residual = pinn.pde_residual(output, input)
        pde_loss = criterion(pde_residual, torch.zeros_like(pde_residual))

        # boundary loss
        bc_loss = 0
        bc_loss_list = []

        # inlet
        bc_residual = pinn.boundary_residual(output, input, type = "inlet", index = index_boundaries[0])
        bc_loss_list.append(criterion(bc_residual, torch.zeros_like(bc_residual)))
        
        # outlet
        bc_residual = pinn.boundary_residual(output, input, type = "outlet", index = index_boundaries[1])
        bc_loss_list.append(criterion(bc_residual, torch.zeros_like(bc_residual)))
        
        # top
        bc_residual = pinn.boundary_residual(output, input, type = "wall", index = index_boundaries[2])
        bc_loss_list.append(criterion(bc_residual, torch.zeros_like(bc_residual)))
        
        # bottom
        bc_residual = pinn.boundary_residual(output, input, type = "wall", index = index_boundaries[3])
        bc_loss_list.append(criterion(bc_residual, torch.zeros_like(bc_residual)))

        # cylinder
        bc_residual = pinn.boundary_residual(output, input, type = "wall", index = index_boundaries[4])
        bc_loss_list.append(criterion(bc_residual, torch.zeros_like(bc_residual)))

        bc_loss = sum(bc_loss_list)
        loss = 1.0 * pde_loss + beta * bc_loss

        loss_log.append(loss.cpu().detach().numpy())

        if epoch % 100 == 0:
            print(
                'epoch:{:05d}, pde_loss: {:.08e}, bc_loss: {:.08e}, loss: {:.08e}'.format(
                    epoch, pde_loss.item(), bc_loss.item(), loss.item()
                )
            )
            writer.add_scalar(path + '/pde_loss', pde_loss, epoch)
            writer.add_scalar(path + '/bc_loss', bc_loss, epoch)
            writer.add_scalar(path + '/loss', loss, epoch)

            plt.clf()
            vis.plot_loss(np.arange(len(loss_log)), np.array(loss_log), "pde_loss")

            plt.savefig(os.path.join(path, 'loss.jpg'))

        if epoch % 1000 == 0:
            torch.save(pinn.state_dict(), path + "model_" + str(num) + ".pth")
            vis.plot_uvp(pinn, True, path, epoch)

        loss.backward()
        optimizer.step()


def test():
    name = "task3_" + str(num)
    path = "save/" + name + "/"

    pinn = PINN([2, 40, 40, 40, 40, 40, 40, 40, 40, 3]).to(device) # x, y -> u, v, p
    pinn.load_state_dict(torch.load(path + "model_" + str(num) + ".pth"))
    _, ground_true, _ = pinn.read_data()
    ground_true = torch.tensor(ground_true, dtype=torch.float32)
    triang = matplotlib.tri.Triangulation(ground_true[:, 0], ground_true[:, 1])
    triang.set_mask(np.hypot(ground_true[triang.triangles, 0].mean(axis=1) - 0.2,
                             ground_true[triang.triangles, 1].mean(axis=1) - 0.2) < r)

    vis = visualise()
    #vis.plot_gt_uvp(ground_true, triang, True, path, 0)
    vis.plot_p_vector(pinn, True, path, 0)