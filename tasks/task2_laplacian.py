from net import Net
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import grad
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import gif

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')


class Plot():
    def __init__(self) -> None:
        self.lr = 1e-3
        self.n_pred = 30
        self.n_f = 30
        self.epochs = 160000
        self.epoch = 0

        self.PINN = Net([2, 64, 64, 64, 64, 1]).to(device)

        self.range_left, self.range_right = -10, 0
        self.optimizer = torch.optim.Adam(self.PINN.parameters(), self.lr)
        self.criterion = torch.nn.MSELoss()

        self.loss_history = []

        # self.figure = plt.figure()
        # self.ax = Axes3D(self.figure)
        # self.ax.set_xlim(self.range_left, self.range_right)
        # self.ax.set_ylim(-4, 4)

        # self.update(0)

    def d(self, f, x):
        return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

    def PDE(self, u, x, y):
        # return d(d(u, x), x) + 0.49 * torch.sin(0.7 * x) + 2.25 * torch.cos(1.5 * x)
        return self.d(self.d(u, x), x) + self.d(self.d(u, y), y)

    def Ground_true(self, x, y):
        # return torch.sin(0.7 * x) + torch.cos(1.5 * x) - 0.1 * x
        return torch.exp(x) * torch.sin(y)
    
    def mesh_generate(self, range_left, range_right, n, type = "rand", grad = True):
        if type == "rand":
            if grad == True:
                x = ((range_left + range_right) / 2 + (range_right - range_left) *
                        (torch.rand(size=(n, 1), dtype=torch.float, device=device) - 0.5)
                        ).requires_grad_(True)
                y = ((range_left + range_right) / 2 + (range_right - range_left) *
                        (torch.rand(size=(n, 1), dtype=torch.float, device=device) - 0.5)
                        ).requires_grad_(True)
                x, y = torch.meshgrid(x.squeeze(1), y.squeeze(1))
            elif grad != True:
                x = ((range_left + range_right) / 2 + (range_right - range_left) *
                        (torch.rand(size=(n, 1), dtype=torch.float, device=device) - 0.5)
                        )
                y = ((range_left + range_right) / 2 + (range_right - range_left) *
                        (torch.rand(size=(n, 1), dtype=torch.float, device=device) - 0.5)
                        )
                x, y = torch.meshgrid(x.squeeze(1), y.squeeze(1))

        else:
            x = torch.linspace(range_left, range_right, n).reshape((-1, 1)).to(device)
            y = torch.linspace(range_left, range_right, n).reshape((-1, 1)).to(device)
            x, y = torch.meshgrid(x.squeeze(1), y.squeeze(1))

        return x, y, torch.cat((x.reshape(-1).unsqueeze(1), y.reshape(-1).unsqueeze(1)), dim=1)

    def boundary_generate(self, range_left, range_right, n, init_val = 0):
        y = ((range_left + range_right) / 2 + (range_right - range_left) *
                (torch.rand(size=(n, 1), dtype=torch.float, device=device) - 0.5)
                ).requires_grad_(True)
        x = init_val * torch.ones_like(y).requires_grad_(True).to(device)

        return x, y, torch.cat((x, y), dim=1)
    
    def boundary_generate_y(self, range_left, range_right, n, init_val = 0):
        x = ((range_left + range_right) / 2 + (range_right - range_left) *
                (torch.rand(size=(n, 1), dtype=torch.float, device=device) - 0.5)
                ).requires_grad_(True)
        y = init_val * torch.ones_like(x).requires_grad_(True).to(device)

        return x, y, torch.cat((x, y), dim=1)
    
    def init(self):
        return self.update(0)
    
    def update(self, frame):
        if self.epoch <= self.epochs:
            self.epoch += 1
            self.optimizer.zero_grad()
            # inside
            x_f, y_f, xy_f = self.mesh_generate(self.range_left, self.range_right, self.n_f)
            u_f = self.PINN(xy_f).reshape(self.n_f, -1)
            PDE_ = self.PDE(u_f, x_f, y_f)
            mse_PDE = self.criterion(PDE_, torch.zeros_like(PDE_))

            # boundary
            # x_b = torch.tensor([[-10.], [0.]]).requires_grad_(True).to(device)
            # y_b = torch.tensor([[0.], [0.]]).requires_grad_(True).to(device)
            # u_b = self.PINN(torch.cat((x_b, y_b), dim=1))
            # true_b = self.Ground_true(x_b, y_b)
            # mse_BC = self.criterion(u_b, true_b)

            x_b2, y_b2, xy_b2 = self.boundary_generate(self.range_left, self.range_right, self.n_f)
            u_b2 = self.PINN(xy_b2)
            true_b2 = self.Ground_true(x_b2, y_b2)
            mse_BC = self.criterion(u_b2, true_b2)

            x_b3, y_b3, xy_b3 = self.boundary_generate(self.range_left, self.range_right, self.n_f, -10)
            u_b3 = self.PINN(xy_b3)
            true_b3 = self.Ground_true(x_b3, y_b3)
            mse_BC += self.criterion(u_b3, true_b3)

            x_b4, y_b4, xy_b4 = self.boundary_generate_y(self.range_left, self.range_right, self.n_f, -10)
            u_b4 = self.PINN(xy_b4)
            true_b4 = self.Ground_true(x_b4, y_b4)
            mse_BC += self.criterion(u_b4, true_b4)

            # y_b5, x_b5, xy_b5 = self.boundary_generate(self.range_left, self.range_right, self.n_f)
            # u_b5 = self.PINN(xy_b5)
            # true_b5 = self.Ground_true(x_b5, y_b5)
            # mse_BC += self.criterion(u_b5, true_b5)

            loss = 1 * mse_PDE + 1 * mse_BC

            if self.epoch % 100 == 0:
                print(
                    'epoch:{:05d}, EoM: {:.08e}, BC: {:.08e}, loss: {:.08e}'.format(
                        self.epoch, mse_PDE.item(), mse_BC.item(), loss.item()
                    )
                )

            loss.backward()
            self.optimizer.step()

            yy = torch.linspace(self.range_left, self.range_right, 2000).reshape((-1, 1)).to(device)
            xx = torch.ones_like(yy)

            # if (self.epoch) % 100 == 0:

            # zz = self.PINN(torch.cat((xx, yy), dim=1))
            # gt = self.Ground_true(xx, yy)
            # xx = xx.reshape((-1)).data.detach().cpu().numpy()
            # yy = yy.reshape((-1)).data.detach().cpu().numpy()
            # zz = zz.reshape((-1)).data.detach().cpu().numpy()
            # gt = gt.reshape((-1)).data.detach().cpu().numpy()

            # self.line1.set_data(yy, zz)
            # self.line2.set_data(yy, gt)

            # if (self.epoch) % 100 == 0:
            n_3d = 1000
            xx, yy, xy = self.mesh_generate(self.range_left, self.range_right, n_3d, type="else")
            zz = self.PINN(xy).reshape(n_3d, -1)
            gt = self.Ground_true(xx, yy)

            xx = xx.data.detach().cpu().numpy()
            yy = yy.data.detach().cpu().numpy()
            zz = zz.data.detach().cpu().numpy()
            gt = gt.data.detach().cpu().numpy()
             
        return xx, yy, zz, gt

gif.options.matplotlib["dpi"] = 300

@gif.frame
def plot_surface_func(ax, xx, yy, zz, gt):
    ax.plot_surface(xx, yy, zz, cmap = "pink")
    ax.plot_surface(xx, yy, gt, cmap = "bone")

    return ax

def make_gif():
    cls_plt = Plot()
    # plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    frames = []
    for i in range(30):
        # for j in range(2):
        #     xx, yy, zz, gt = cls_plt.update(0)
        xx, yy, zz, gt = cls_plt.update(0)
        fram = plot_surface_func(ax, xx, yy, zz, gt)
        # if i %2 ==0:
        frames.append(fram)
        # plt.pause(0.01)

    gif.save(frames,'tu1.gif', duration=3.5, unit="s", between="startend")

def train():
    cls_plt = Plot()
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10000):
        ax.cla()
        for j in range(20):
            xx, yy, zz, gt = cls_plt.update(0)
        ax.plot_surface(xx, yy, zz, cmap = "pink", alpha=0.9)
        ax.plot_surface(xx, yy, gt, cmap = "bone", alpha=0.6)
        # ax.plot_surface(xx, yy, (zz - gt), cmap = "pink")
        plt.pause(0.01)

    plt.ioff()
    plt.show()    

if __name__ == "__main__":
    train()