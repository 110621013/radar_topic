'''資料部分：
小端存儲
空間：115.0°E~126.5°E，18.0°N~29°N，delta 0.0125°
時間：06:00Z09Jun2020 開始10分鐘一筆共19筆 (到09:00Z09Jun2020)
兩個變量：dbz和rr
'''
# 工作環境：中央大氣server140.115.35.156, pwd: /home/classSTS/2021_AP3111/NCDR_maple/106601015

import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.colors import from_levels_and_colors, ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import imageio

grd_path = os.path.join('..')
x_num, y_num, t_num, v_num = 921, 881, 19, 2
lonlat_delta = 0.0125
lon_upper, lon_lower, lat_upper, lat_lower = 126.5, 115, 29, 18
rr_colors = [
    "#fdfdfd", #white
    "#ccccb3", #grays
    "#99ffff", #blue
    "#66b3ff",
    "#0073e6",
    "#002699",
    "#009900", #green
    "#00c900",
    "#1aff1a",
    "#ffff00", #yellow
    "#ffcc00",
    "#ff9900",
    "#ff0000", #red
    "#cc0000",
    "#990000",
    "#800040", #purple
    "#b30047",
    "#ff0066",
    "#ff80b3",
]
dbz_colors = [
    "#99ffff", #blue
    "#66b3ff",
    "#0073e6",
    "#002699",
    "#009900", #green
    "#1aff1a",
    "#ffff00", #yellow
    "#ff9900",
    "#ff0000", #red
    "#cc0000",
    "#800040", #purple
    "#b30047",
    "#ff2eff",
]
delta_colors = [ # deep blue - white - deep red
    "#000079",
    "#0000C6",
    "#2828FF",
    "#6A6AFF",
    "#9393FF",
    "#B9B9FF",
    "#FFFFFF",
    "#FFB5B5",
    "#FF7575",
    "#FF2D2D",
    "#FF0000",
    "#CE0000",
    "#750000",
]


# arg: path+filename, return nparray(t, var, y, x)
def read_grd(filename):
    data = np.fromfile(filename, dtype="<f4")
    data = data.reshape((t_num, v_num, y_num, x_num), order='C')
    return data
# check all folder we need
def check_output_folder():
    check_folder_list = [
        os.path.join('.', 'data_output_img'),
        os.path.join('.', 'data_output_img', 'dbz'),
        os.path.join('.', 'data_output_img', 'rr'),
        os.path.join('.', 'data_output_img', 'RMSE'),
        os.path.join('.', 'data_output_img', 'k_mean'),
    ]
    for folder in check_folder_list:
        if os.path.exists(folder) == False:
            os.mkdir(folder)


# arg: dbzname, dbzfield, savepath, make contourf image
def plot_basic_contourf(dataname, datafield, title, savepath, savename):
    fig = plt.figure()
    # set base map and lon/lat
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', alpha=0.7)
    ax.set_title(title)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xlim([lon_lower, lon_upper])
    ax.set_ylim([lat_lower, lat_upper])
    x = np.arange(lon_lower, lon_upper+lonlat_delta/2, lonlat_delta) # <---strange ,no add delta->920/add delta->922
    y = np.arange(lat_lower, lat_upper+lonlat_delta, lonlat_delta)
    X, Y = np.meshgrid(x, y)

    # set grid line and colormap
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='gray', alpha=0.4)
    gl.xlocator = mticker.FixedLocator(np.arange(lon_lower, lon_upper, 1)) #grid line set
    gl.ylocator = mticker.FixedLocator(np.arange(lat_lower, lat_upper, 1))
    gl.top_labels = False
    gl.right_labels = False

    # plot part
    if dataname == 'dbz':
        levels = range(0, 65, 5)
        norm = BoundaryNorm(levels, 12)
        dbz_colormap = ListedColormap(dbz_colors)
        plot = ax.contourf(X, Y, datafield, extend='max', levels=levels, norm=norm, cmap=dbz_colormap)
    elif dataname == 'rr':
        levels = [0,0.5,1,2,6,10,15,20,30,40,50,70,90,110,130,150,200,300,400]
        norm = BoundaryNorm(levels, 18)
        rr_colormap = ListedColormap(rr_colors)
        plot = ax.contourf(X, Y, datafield, extend='max', levels=levels, norm=norm, cmap=rr_colormap)
    elif dataname == 'delta_dbz':
        levels = range(-65, 65, 10)
        norm = BoundaryNorm(levels, 12)
        delta_colormap = ListedColormap(delta_colors)
        plot = ax.contourf(X, Y, datafield, extend='max', levels=levels, norm=norm, cmap=delta_colormap)
    cbar = plt.colorbar(plot)
    cbar.set_label(dataname, rotation=0, labelpad=-25, y=1.05)
    cbar.set_ticks(levels)

    plt.savefig(os.path.join(savepath, savename))
    plt.close()

#arg: 3D t-x-y data, plot + save + make gif
def plot_data_sequentially_contourf(dataname, datafield, data_time, delete_png_flag):
    savepath = os.path.join('.', 'data_output_img', dataname)
    plot_img_list = []
    for i in range(datafield.shape[0]):
        title = data_time[:-4]+'-'+data_time[-4:]+'  '+str(i*10)+'mins after'
        savename = '{}-{}_{}_{}'.format(data_time[:-4], data_time[-4:], dataname, str(i*10)) + '.png'
        plot_basic_contourf(dataname, datafield[i], title, savepath, savename)
        plot_img_list.append(os.path.join(savepath, savename))

    # make gif
    img_list = []
    for img_path in plot_img_list:
        img_list.append(imageio.imread(img_path))
    imageio.mimsave(os.path.join(savepath, '{}-{}_{}.gif'.format(data_time[:-4], data_time[-4:], dataname)), img_list)

    # remove png img
    if delete_png_flag:
        for png in plot_img_list:
            os.remove(png)

#arg: forecasts field, observations field, time_interval, calculate RMSE and plot deltafield
def plot_deltafield_RMSE(data, data_bar, time_interval_name):
    savepath = os.path.join('.', 'data_output_img', 'RMSE')
    savename = time_interval_name
    # calculate delta_dbz_field and RMSE
    delta_dbz_field = data-data_bar
    rmse = math.sqrt(np.sum(np.power(delta_dbz_field, 2)) / np.size(delta_dbz_field))

    # plot part
    title = time_interval_name+' delta dbz field, RMSE:{:.4f}'.format(rmse)
    plot_basic_contourf('delta_dbz', delta_dbz_field, title, savepath, savename)


############################################################################ TODO: dbz plot

# calculate (x, y)<->(kx, ky) distance
def dis(x, y, kx, ky):
    return int(((kx-x)**2 + (ky-y)**2)**0.5)
# k-means grouping
def kmeans(x, y, kx, ky, dot_num, seed_num, con, fig):
    # grouping each element
    group = []
    for i in range(seed_num):
        group.append([])
    min_dis = 99999999
    for i in range(dot_num):
        for j in range(seed_num):
            distant = dis(x[i], y[i], kx[j], ky[j])
            if distant < min_dis:
                min_dis = distant
                flag = j
        group[flag].append([x[i], y[i]])
        min_dis = 99999999
    # find the new grouping center for the grouped elements
    sumx, sumy = 0, 0
    nkx, nky = [], []
    for index, nodes in enumerate(group):
        if nodes == []: # if no node in this group, pass away
            nkx.append(kx[index])
            nky.append(ky[index])
        else:
            for node in nodes:
                sumx += node[0]
                sumy += node[1]
            nkx.append(int(sumx/len(nodes)))
            nky.append(int(sumy/len(nodes)))
            sumx, sumy = 0, 0

    # plot kmean point
    save_path = os.path.join('.', 'data_output_img', 'k_mean')
    cx, cy = [], []
    line = plt.gca()
    line.set_xlim([0, x_num])
    line.set_ylim([0, y_num])
    for index, nodes in enumerate(group):
        for node in nodes:
            cx.append([node[0], nkx[index]])
            cy.append([node[1], nky[index]])
        for i in range(len(cx)):
            line.plot(cx[i], cy[i], color='r', alpha=0.5)
        cx = []
        cy = []
    feature = plt.scatter(x, y, s=50)
    #k_feature = plt.scatter(kx, ky, s=50)
    cbar = plt.colorbar(feature) # <--just for alignment
    nk_feaure = plt.scatter(np.array(nkx), np.array(nky), s=50)
    plt.savefig(os.path.join(save_path, 'kmeans_{}_t{}.png'.format(str(con), str(fig))))
    plt.close()

    # determine whether the grouping center is no longer changed
    if nkx == list(kx) and nky == list(ky):
        return fig
    else:
        fig += 1
        fig = kmeans(x, y, nkx, nky, dot_num, seed_num, con, fig)
    return fig



# plot and save all img from grd_path
def create_NCDR_maple_img(delete_png_flag):
    grd_list = [x for x in os.listdir(grd_path) if x.split('.')[-1] =='grd']
    grd_list.sort()

    # for all grd data, read and get dbz, rr
    for grd_name in grd_list:
        grd_abs_filename = os.path.join(grd_path, grd_name)
        tvyx_arr = read_grd(grd_abs_filename)
        dbz, rr = tvyx_arr[:, 0, :, :], tvyx_arr[:, 1, :, :]
        print('---> grd_abs_filename:', grd_abs_filename)

        # data pretreatment, dbz:(0~65), rr:(1~300)
        dbz[dbz <= 0], rr[rr <= 1] = np.nan, np.nan
        dbz[dbz >= 65], rr[rr >=300] = np.nan, np.nan
        plot_data_sequentially_contourf('dbz', dbz, grd_name[-16:-4], delete_png_flag=delete_png_flag)
        plot_data_sequentially_contourf('rr', rr, grd_name[-16:-4], delete_png_flag=delete_png_flag)

# compare forecasts with observations (specialized reading)
def compare_forecasts_effectiveness():
    # read grd
    dbz_21_12_obs = read_grd(os.path.join(grd_path, 'fstdbz_202011211200.grd'))[0, 0, :, :]
    dbz_21_11_1h = read_grd(os.path.join(grd_path, 'fstdbz_202011211100.grd'))[6, 0, :, :]
    dbz_21_10_2h = read_grd(os.path.join(grd_path, 'fstdbz_202011211000.grd'))[12, 0, :, :]
    # data filter
    dbz_21_12_obs[dbz_21_12_obs <= 0], dbz_21_11_1h[dbz_21_11_1h <= 0], dbz_21_10_2h[dbz_21_10_2h <= 0] = 0.0, 0.0, 0.0
    dbz_21_12_obs[dbz_21_12_obs >= 65], dbz_21_11_1h[dbz_21_11_1h >= 65], dbz_21_10_2h[dbz_21_10_2h >= 65] = 0.0, 0.0, 0.0
    # plot deltafield and mark RMSE
    plot_deltafield_RMSE(dbz_21_11_1h, dbz_21_12_obs, '1hour')
    plot_deltafield_RMSE(dbz_21_10_2h, dbz_21_12_obs, '2hour')

# do k-mean for all files and plot
def k_mean_convectivecell_marking():
    seed_num, con = 10, 0
    kx, ky = np.random.randint(0, x_num, seed_num), np.random.randint(0, y_num, seed_num)
    ####################

    # read grd
    dbz_21_12_obs = read_grd(os.path.join(grd_path, 'fstdbz_202011211200.grd'))[0, 0, :, :]
    dbz_21_11_1h = read_grd(os.path.join(grd_path, 'fstdbz_202011211100.grd'))[6, 0, :, :]
    dbz_21_10_2h = read_grd(os.path.join(grd_path, 'fstdbz_202011211000.grd'))[12, 0, :, :]
    dbz_dict = {
        'dbz_21_12_obs':dbz_21_12_obs,
        'dbz_21_11_1h':dbz_21_11_1h,
        'dbz_21_10_2h':dbz_21_10_2h,
    }

    for dbzname, dbzfield in dbz_dict.items():
        # version1: Find the threshold so that the marking points are >100
        '''
        threshold = 50.0
        # renew threshold and dot_num
        while True:
            x, y = [], []
            for j in range(dbzfield.shape[0]):
                for i in range(dbzfield.shape[1]):
                    if dbzfield[j, i]<65.0 and dbzfield[j, i]>threshold:
                        x.append(i)
                        y.append(j)
            dot_num = len(x)

            if dot_num < 100:
                threshold -= 1.0
                print('----> threshold, dot_num:', threshold, dot_num)
            else:
                break
        '''
        # version2: threshold -> dbzfield>0
        threshold = 40.0 ####################
        x, y = [], []
        for j in range(dbzfield.shape[0]):
            for i in range(dbzfield.shape[1]):
                if dbzfield[j, i]<65.0 and dbzfield[j, i]>threshold:
                    x.append(i)
                    y.append(j)
        dot_num = len(x)

        print('---> threshold, dot_num, seed_num:', threshold, dot_num, seed_num)
        savepath = os.path.join('.', 'data_output_img', 'k_mean')

        # clean old data
        for root, dirs, files in os.walk(savepath):
            for name in files:
                os.remove(os.path.join(root, name))

        # plot basic dbz image to overlay
        title = 'dbz {} to overlay'.format(str(con))
        savename = 'dbz{}'.format(str(con))
        plot_basic_contourf('dbz', dbzfield, title, savepath, savename)
        # call kmeans to plot
        final_fig = kmeans(x, y, kx, ky, dot_num=dot_num, seed_num=seed_num, con=con, fig=0)

        # overlay
        dbz_layer = plt.imread(os.path.join(savepath, savename+'.png'))
        k_layer = plt.imread(os.path.join(savepath, 'kmeans_{}_t{}.png'.format(str(con), str(final_fig)) ))
        plt.imshow(dbz_layer, alpha=0.5)
        plt.imshow(k_layer, alpha=0.5)

        plt.savefig(os.path.join(savepath, 'overlay_{}'.format(str(con))))
        plt.close()

        con += 1

# print pearson score (10+2 vs 12 / 11+1 vs 12)
def pearson():
    print('----> pearson gogo')
    # read grd
    dbz_21_12_obs = read_grd(os.path.join(grd_path, 'fstdbz_202011211200.grd'))[0, 0, :, :]
    dbz_21_11_1h = read_grd(os.path.join(grd_path, 'fstdbz_202011211100.grd'))[6, 0, :, :]
    dbz_21_10_2h = read_grd(os.path.join(grd_path, 'fstdbz_202011211000.grd'))[12, 0, :, :]
    mean_obs = dbz_21_12_obs.mean()
    mean_1h = dbz_21_11_1h.mean()
    mean_2h = dbz_21_10_2h.mean()

    up, down1, down2 = 0, 0, 0
    for j in range(y_num) :
        for i in range(x_num) :
            up += (dbz_21_11_1h[j,i]-mean_1h)*(dbz_21_12_obs[j,i]-mean_obs)
            down1 += (dbz_21_11_1h[j,i]-mean_1h)**2
            down2 += (dbz_21_12_obs[j,i]-mean_obs)**2
    pearson_dbz_1h = up/(((down1)**0.5)*((down2)**0.5))

    up, down1, down2 = 0, 0, 0
    for j in range(y_num) :
        for i in range(x_num) :
            up += (dbz_21_10_2h[j,i]-mean_2h)*(dbz_21_12_obs[j,i]-mean_obs)
            down1 += (dbz_21_10_2h[j,i]-mean_2h)**2
            down2 += (dbz_21_12_obs[j,i]-mean_obs)**2
    pearson_dbz_2h = up/(((down1)**0.5)*((down2)**0.5))


    rr_21_12_obs = read_grd(os.path.join(grd_path, 'fstdbz_202011211200.grd'))[0, 1, :, :]
    rr_21_11_1h = read_grd(os.path.join(grd_path, 'fstdbz_202011211100.grd'))[6, 1, :, :]
    rr_21_10_2h = read_grd(os.path.join(grd_path, 'fstdbz_202011211000.grd'))[12, 1, :, :]
    mean_obs = rr_21_12_obs.mean()
    mean_1h = rr_21_11_1h.mean()
    mean_2h = rr_21_10_2h.mean()

    up, down1, down2 = 0, 0, 0
    for j in range(y_num) :
        for i in range(x_num) :
            up += (rr_21_11_1h[j,i]-mean_1h)*(rr_21_12_obs[j,i]-mean_obs)
            down1 += (rr_21_11_1h[j,i]-mean_1h)**2
            down2 += (rr_21_12_obs[j,i]-mean_obs)**2
    pearson_rr_1h = up/(((down1)**0.5)*((down2)**0.5))

    up, down1, down2 = 0, 0, 0
    for j in range(y_num) :
        for i in range(x_num) :
            up += (rr_21_10_2h[j,i]-mean_2h)*(rr_21_12_obs[j,i]-mean_obs)
            down1 += (rr_21_10_2h[j,i]-mean_2h)**2
            down2 += (rr_21_12_obs[j,i]-mean_obs)**2
    pearson_rr_2h = up/(((down1)**0.5)*((down2)**0.5))

    print('pearson_dbz_1h:', pearson_dbz_1h)
    print('pearson_dbz_2h:', pearson_dbz_2h)
    print('pearson_rr_1h:', pearson_rr_1h)
    print('pearson_rr_2h:', pearson_rr_2h)

# print moment score (10+2 vs 12 / 11+1 vs 12)
def moment():
    print('----> moment gogo')
    print('等憶彤code弄好')



if __name__ == '__main__':
    # data_source : ..
    check_output_folder()

    #create_NCDR_maple_img(delete_png_flag=False)
    #compare_forecasts_effectiveness()

    #k_mean_convectivecell_marking()

    pearson()
    moment()

    # TODO
    # 0.code整合上github
    # 1.完善kmean的數個可改進方向()
    # 2.完成線性外延（期待有產出）
    # 3.深度學習資料確認與convLSTM模型初探