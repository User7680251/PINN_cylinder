import torch
import numpy as np
import matplotlib.pyplot as plt


# 定义问题相关参数
area = [0, 1.1, 0, 0.41] # xmin, xmax, ymin, ymax
r = 0.05 # 圆柱半径
pos_x, pos_y = 0.15 + r, 0.15 + r

# 定义训练相关参数
num = 7
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')


class visualise():
    def __init__(self) -> None:
        self.level = None
        n_level = 200
        self.level_p = np.linspace(-0.1, 3.6, n_level)
        self.level_u = np.linspace(0, 1.3, n_level)
        self.level_v = np.linspace(-0.6, 0.6, n_level)

    def plot_loss(self, x, y, label, title=None):
        plt.plot(x, y, label=label)
        plt.semilogy()
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(title)

    def process_data(self, nx, ny, pinn, area=area):
        x = torch.linspace(area[0], area[1], nx).reshape((-1, 1)).to(device)
        y = torch.linspace(area[2], area[3], ny).reshape((-1, 1)).to(device)
        x, y = torch.meshgrid(x.squeeze(1), y.squeeze(1))
        mesh = torch.cat((x.reshape(-1).unsqueeze(1), y.reshape(-1).unsqueeze(1)), dim=1)

        temp = pinn(mesh)
        u, v, p = temp[:, 0].reshape(nx, ny, -1), temp[:, 1].reshape(nx, ny, -1), temp[:, 2].reshape(nx, ny, -1)
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        p = p.detach().cpu().numpy()

        u = u[:,:,0]
        v = v[:,:,0]
        p = p[:,:,0]

        x = torch.linspace(area[0], area[1], nx).reshape((-1, 1))
        y = torch.linspace(area[2], area[3], ny).reshape((-1, 1))
        x, y = torch.meshgrid(x.squeeze(1), y.squeeze(1))
        x, y = x.numpy(), y.numpy()

        return u, v, p, x, y

    def plot_uvp(self, pinn, save_fig:bool, path:str, epoch:int):
        """
        绘制流线图函数
        
        :param pinn: 网络
        :param save_fig: 保存图片或显示
        :param path: 保存图片路径
        :param epoch: epoch数
        """

        # 网络输出
        nx, ny = 28, 10
        scale = 4
        nx, ny = nx * scale, ny * scale
        area = [0, 1.1, 0, 0.41]
        if self.level is None:
            n_levels = 200
        else:
            n_levels = self.level
        
        u, v, p, x, y = self.process_data(nx, ny, pinn)
        
        #进行绘图
        fig = plt.figure(dpi=400,figsize=(7,8))

        ax1 = plt.subplot(3, 1, 1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Pressure P (Pa)", fontsize=12)
        ax1.set_aspect(1)
        var = p
        cs = ax1.contourf(x, y, var, self.level_p, cmap=plt.get_cmap('RdBu_r'))
        ax1.figure.colorbar(cs, ax=ax1, aspect = 20)
        circle = plt.Circle((pos_x, pos_y), r, color='gray')
        ax1.add_artist(circle)

        ax2 = plt.subplot(3, 1, 2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("X_Velocity u (m/s)", fontsize=12)
        ax2.set_aspect(1)
        var = u
        cs = ax2.contourf(x, y, var, self.level_u, cmap=plt.get_cmap('RdBu_r'))
        ax2.figure.colorbar(cs, ax=ax2, aspect = 20)
        circle = plt.Circle((pos_x, pos_y), r, color='gray')
        ax2.add_artist(circle)

        ax3 = plt.subplot(3, 1, 3)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Y_Velocity v (m/s)", fontsize=12)
        ax3.set_aspect(1)
        var = v
        cs = ax3.contourf(x, y, var, self.level_v, cmap=plt.get_cmap('RdBu_r'))
        ax3.figure.colorbar(cs, ax=ax3, aspect = 20)
        circle = plt.Circle((pos_x, pos_y), r, color='gray')
        ax3.add_artist(circle)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)

        if save_fig:
            img_path = path + "epoch" + str(epoch) + ".jpg"
            plt.savefig(img_path, dpi = 400)
        else:
            plt.show()

    def plot_gt_uvp(self, gt, triang, save_fig:bool, path:str, epoch:int):
        """
        绘制流线图函数
        
        :param pinn: 网络
        :param save_fig: 保存图片或显示
        :param path: 保存图片路径
        :param epoch: epoch数
        """

        # 网络输出
        nx, ny = 28, 10
        scale = 4
        nx, ny = nx * scale, ny * scale
        area = [0, 1.1, 0, 0.41]
        n_levels = 40

        gt = gt[..., 2:].cpu().numpy()
        self.level = np.arange(gt.min(), gt.max(), 0.05)
        
        #进行绘图
        fig = plt.figure(dpi=400,figsize=(7,8))

        ax1 = plt.subplot(3, 1, 1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("CFD Pressure P (Pa)", fontsize=12)
        ax1.set_aspect(1)
        cs = ax1.tricontourf(triang, gt[:, 0], self.level_p, cmap=plt.get_cmap('RdBu_r'))
        ax1.figure.colorbar(cs, ax=ax1, aspect = 20)
        circle = plt.Circle((pos_x, pos_y), r, color='gray')
        ax1.add_artist(circle)

        ax2 = plt.subplot(3, 1, 2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("CFD X_Velocity u (m/s)", fontsize=12)
        ax2.set_aspect(1)
        cs = ax2.tricontourf(triang, gt[:, 1], self.level_u, cmap=plt.get_cmap('RdBu_r'))
        ax2.figure.colorbar(cs, ax=ax2, aspect = 20)
        circle = plt.Circle((pos_x, pos_y), r, color='gray')
        ax2.add_artist(circle)

        ax3 = plt.subplot(3, 1, 3)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("CFD Y_Velocity v (m/s)", fontsize=12)
        ax3.set_aspect(1)
        cs = ax3.tricontourf(triang, gt[:, 2], self.level_v, cmap=plt.get_cmap('RdBu_r'))
        ax3.figure.colorbar(cs, ax=ax3, aspect = 20)
        circle = plt.Circle((pos_x, pos_y), r, color='gray')
        ax3.add_artist(circle)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)

        if save_fig:
            img_path = path + "epoch" + str(epoch) + ".jpg"
            plt.savefig(img_path, dpi = 400)
        else:
            plt.show()

    def plot_p_vector(self, pinn, save_fig:bool, path:str, epoch:int):
        """
        绘制流线图函数
        
        :param pinn: 网络
        :param save_fig: 保存图片或显示
        :param path: 保存图片路径
        :param epoch: epoch数
        """

        # 网络输出
        nx, ny = 35, 15
        scale = 1
        nx, ny = nx * scale, ny * scale
        area = [0, 1.1, 0, 0.41]
        if self.level is None:
            n_levels = 200
        else:
            n_levels = self.level
        
        u, v, p, x, y = self.process_data(nx, ny, pinn)
        
        #进行绘图
        # fig = plt.figure(dpi=400,figsize=(7,8))

        # ax1 = plt.subplot()
        fig, ax1 = plt.subplots()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Pressure with Velocity", fontsize=12)
        ax1.set_aspect(1)
        var = p
        cs = ax1.contourf(x, y, var, self.level_p, cmap=plt.get_cmap('RdBu_r'))
        circle = plt.Circle((pos_x, pos_y), r, color='gray')
        ax1.quiver(x, y, u, v, alpha=0.8, color="white")
        ax1.add_artist(circle)
        ax1.figure.colorbar(cs, ax=ax1, aspect = 20, shrink = 0.4)

        if save_fig:
            img_path = path + "epoch" + str(epoch) + ".jpg"
            plt.savefig(img_path, dpi = 400)
        else:
            plt.show()

    def plot_stream(self, pinn, save_fig:bool, path:str, epoch:int):
        """
        绘制流线图函数
        
        :param pinn: 网络
        :param save_fig: 保存图片或显示
        :param path: 保存图片路径
        :param epoch: epoch数
        """

        # 网络输出
        nx, ny = 28, 28
        nx, ny = nx * 4, ny * 4

        area = [0, 0.41, 0, 0.41]
        
        u, v, p, x, y = self.process_data(nx, ny, pinn, area)
        
        #进行绘图
        circle = plt.Circle((pos_x, pos_y), r, color='gray')
        fig1, ax1 = plt.subplots()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Streamline diagram", fontsize=12)
        ax1.set_aspect(1)
        var = p
        n_levels = 200
        cs = ax1.contourf(x, y, var, n_levels, cmap=plt.get_cmap('RdBu_r'))
        ax1.figure.colorbar(cs, ax=ax1, aspect = 20)
        #ax1.quiver(x,y,u,v,alpha=0.8)
        ax1.streamplot(y,x,u,v,color='white', zorder=1, density=2, linewidth = 0.7)
        ax1.add_artist(circle)

        if save_fig:
            img_path = path + "epoch" + str(epoch) + ".jpg"
            plt.savefig(img_path, dpi = 200)
        else:
            plt.show()
