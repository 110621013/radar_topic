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
fig = 0
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
score_colors = [
    '#FF0000',
    '#FF0000',
    '#28FF28',
    '#28FF28',
    '#009999A9',
    '#009999A9',
    '#CEFFCd',
    '#CEFFCd',
]


# arg: path+filename, return data nparray(t, var, y, x)
def read_grd(filename):
    data = np.fromfile(filename, dtype="<f4")
    data = data.reshape((t_num, v_num, y_num, x_num), order='C')
    return data
# arg: data nparray(y, x) + check threshold + noise delta, return fixed data nparray(y, x)
def filterout_localnoise(data, threshold, delta):
    for j in range(data.shape[0]):
        for i in range(data.shape[1]):
            if data[j, i] >= threshold:
                is_noise_flag = False
                if j != 0:
                    if abs(data[j, i] - data[j-1, i]) > delta:
                        is_noise_flag = True
                if j != data.shape[0]-1:
                    if abs(data[j, i] - data[j+1, i]) > delta:
                        is_noise_flag = True
                if i != 0:
                    if abs(data[j, i] - data[j, i-1]) > delta:
                        is_noise_flag = True
                if i != data.shape[1]-1:
                    if abs(data[j, i] - data[j, i+1]) > delta:
                        is_noise_flag = True

                if is_noise_flag:
                    data[j, i] = 0.0
                    con = 0
                    if j != 0:
                        data[j, i] += data[j-1, i]
                        con += 1
                    if j != data.shape[0]-1:
                        data[j, i] += data[j+1, i]
                        con += 1
                    if i != 0:
                        data[j, i] += data[j, i-1]
                        con += 1
                    if i != data.shape[1]-1:
                        data[j, i] += data[j, i+1]
                        con += 1
                    data[j, i] /= con
    return data


# check all folder we need
def check_output_folder():
    check_folder_list = [
        os.path.join('.', 'data_output_img'),
        os.path.join('.', 'data_output_img', 'dbz'),
        os.path.join('.', 'data_output_img', 'rr'),
        os.path.join('.', 'data_output_img', 'RMSE'),
        os.path.join('.', 'data_output_img', 'k_mean'),
        os.path.join('.', 'data_output_img', 'k_mean_v2'),
        os.path.join('.', 'data_output_img', 'k_mean_v3'),
        os.path.join('.', 'data_output_img', 'score'),
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
    elif dataname == 'index':
        levels = [-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75]
        norm = BoundaryNorm(levels, 8)
        score_colormap = ListedColormap(score_colors)
        plot = ax.contourf(X, Y, datafield, extend='max', levels=levels, norm=norm, cmap=score_colormap)


    if dataname == 'index':
        cbar = plt.colorbar(plot, ticks=[0, 0.5, 1, 1.5])
        cbar.set_label(dataname, rotation=0, labelpad=-70, y=1.05)
        cbar.ax.set_yticklabels(['hits', 'misses', 'false alarm', 'correct\nnegatives'])
    else:
        cbar = plt.colorbar(plot)
        cbar.set_label(dataname, rotation=0, labelpad=-25, y=1.05)
        cbar.set_ticks(levels)

    # special case
    #if dataname == 'index':
    #    cbar.set_ticks(['hits', 'misses', 'false alarm', 'correct negatives'])

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


# calculate (x, y)<->(kx, ky) distance
def dis(x, y, kx, ky):
    return int(((kx-x)**2 + (ky-y)**2)**0.5)
# k-means grouping
def kmeans(element_list, kernel_list, savepath, savename):
    global fig
    kernel_num = len(kernel_list[0])
    element_num = len(element_list[0])
    # grouping each element
    group = []
    for i in range(kernel_num):
        group.append([])
    min_dis = 99999999
    for i in range(element_num):
        for j in range(kernel_num):
            distant = dis(element_list[0][i], element_list[1][i], kernel_list[0][j], kernel_list[1][j])
            if distant < min_dis:
                min_dis = distant
                flag = j
        group[flag].append([element_list[0][i], element_list[1][i]])
        min_dis = 99999999
    # find the new grouping center for the grouped elements
    sumx, sumy = 0, 0
    new_kernel_list = [[], []]
    for index, nodes in enumerate(group):
        if nodes == []: # if no node in this group, pass away
            new_kernel_list[0].append(kernel_list[0][index])
            new_kernel_list[1].append(kernel_list[1][index])
        else:
            for node in nodes:
                sumx += node[0]
                sumy += node[1]
            new_kernel_list[0].append(int(sumx/len(nodes)))
            new_kernel_list[1].append(int(sumy/len(nodes)))
            sumx, sumy = 0, 0

    # plot kmean point
    line = plt.gca()
    line.set_xlim([0, x_num])
    line.set_ylim([0, y_num])

    line_list = [[], []]
    for index, nodes in enumerate(group):
        for node in nodes:
            line_list[0].append([node[0], new_kernel_list[0][index]])
            line_list[1].append([node[1], new_kernel_list[1][index]])
        for i in range(len(line_list[0])):
            line.plot(line_list[0][i], line_list[1][i], color='r', alpha=0.3)
        line_list = [[], []]
    element = plt.scatter(element_list[0], element_list[1], s=30)
    #k_feature = plt.scatter(kx, ky, s=50)
    plt.colorbar(element) # <--just for alignment
    plt.scatter(np.array(new_kernel_list[0]), np.array(new_kernel_list[1]), s=30)
    plt.savefig(os.path.join(savepath, 'kmeans_{}_t{}.png'.format(savename, str(fig))))
    plt.close()

    # determine whether the grouping center is no longer changed
    if new_kernel_list[0] == list(kernel_list[0]) and new_kernel_list[1] == list(kernel_list[1]):
        return
    else:
        fig += 1
        kmeans(element_list, new_kernel_list, savepath, savename)



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

# do k-mean for all files and plot, version 1(random kernel, fixed threshold)
def k_mean_convectivecell_marking_v1():
    global fig
    kernel_num = 10
    threshold = 40.0
    kernel_list = []
    kernel_list.append(list(np.random.randint(0, x_num, kernel_num)))
    kernel_list.append(list(np.random.randint(0, y_num, kernel_num)))
    savepath = os.path.join('.', 'data_output_img', 'k_mean')

    # clean kmean old_file
    old_file_list = os.listdir(savepath)
    for f in old_file_list:
        os.remove(os.path.join(savepath, f))

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
        # data noise filter
        dbzfield = filterout_localnoise(data=dbzfield, threshold=threshold, delta=threshold)

        element_list = [[], []]
        for j in range(dbzfield.shape[0]):
            for i in range(dbzfield.shape[1]):
                if dbzfield[j, i]<65.0 and dbzfield[j, i]>threshold:
                    element_list[0].append(i)
                    element_list[1].append(j)
        print('---> threshold, element_num, kernel_num:', threshold, len(element_list[0]), kernel_num)

        # plot basic dbz image to overlay
        title = '{} to overlay'.format(dbzname)
        savename = dbzname
        plot_basic_contourf('dbz', dbzfield, title, savepath, savename)
        # call kmeans to plot
        kmeans(element_list, kernel_list, savepath=savepath, savename=savename)

        # overlay
        dbz_layer = plt.imread(os.path.join(savepath, savename+'.png'))
        k_layer = plt.imread(os.path.join(savepath, 'kmeans_{}_t{}.png'.format(savename, str(fig)) ))
        plt.imshow(dbz_layer, alpha=0.5)
        plt.imshow(k_layer, alpha=0.5)

        plt.savefig(os.path.join(savepath, 'overlay_{}'.format(savename)))
        plt.close()
        fig = 0

# do k-mean for all files and plot, version 2(data filter, random + pick only kernel, fixed threshold)
def k_mean_convectivecell_marking_v2():
    global fig
    kernel_num = 10
    threshold = 40.0
    kernel_list = []
    kernel_list.append(list(np.random.randint(0, x_num, kernel_num)))
    kernel_list.append(list(np.random.randint(0, y_num, kernel_num)))
    savepath = os.path.join('.', 'data_output_img', 'k_mean_v2')

    # clean kmean old_file
    old_file_list = os.listdir(savepath)
    for f in old_file_list:
        os.remove(os.path.join(savepath, f))

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
        # data noise filter
        dbzfield = filterout_localnoise(data=dbzfield, threshold=threshold, delta=threshold)

        element_list = [[], []]
        for j in range(dbzfield.shape[0]):
            for i in range(dbzfield.shape[1]):
                if dbzfield[j, i]<65.0 and dbzfield[j, i]>threshold:
                    element_list[0].append(i)
                    element_list[1].append(j)
        print('---> threshold, element_num, kernel_num:', threshold, len(element_list[0]), kernel_num)

        # plot basic dbz image to overlay
        title = '{} to overlay'.format(dbzname)
        savename = dbzname
        plot_basic_contourf('dbz', dbzfield, title, savepath, savename)
        # call kmeans to plot
        kmeans(element_list, kernel_list, savepath=savepath, savename=savename)

        # overlay
        dbz_layer = plt.imread(os.path.join(savepath, savename+'.png'))
        k_layer = plt.imread(os.path.join(savepath, 'kmeans_{}_t{}.png'.format(savename, str(fig)) ))
        plt.imshow(dbz_layer, alpha=0.5)
        plt.imshow(k_layer, alpha=0.5)

        plt.savefig(os.path.join(savepath, 'overlay_{}'.format(savename)))
        plt.close()
        fig = 0



########################################################################

# print pearson score (day 21)(10+2 vs 12 / 11+1 vs 12)
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

# print moment score (day 21)(10+2 vs 12 / 11+1 vs 12)
def moment():
    print('----> moment gogo')
    delta = 1.0
    # read grd
    dbz_21_12_obs = read_grd(os.path.join(grd_path, 'fstdbz_202011211200.grd'))[0, 0, :, :]
    dbz_21_11_1h = read_grd(os.path.join(grd_path, 'fstdbz_202011211100.grd'))[6, 0, :, :]
    dbz_21_10_2h = read_grd(os.path.join(grd_path, 'fstdbz_202011211000.grd'))[12, 0, :, :]
    dbz_21_12_obs[dbz_21_12_obs<0.0] = 0
    dbz_21_11_1h[dbz_21_11_1h<0.0] = 0
    dbz_21_10_2h[dbz_21_10_2h<0.0] = 0
    dbz_21_dict = {
        'dbz_21_10_2h':dbz_21_10_2h,
        'dbz_21_11_1h':dbz_21_11_1h,
        'dbz_21_12_obs':dbz_21_12_obs,
    }
    dbz_23_12_obs = read_grd(os.path.join(grd_path, 'fstdbz_202011231200.grd'))[0, 0, :, :]
    dbz_23_11_1h = read_grd(os.path.join(grd_path, 'fstdbz_202011231100.grd'))[6, 0, :, :]
    dbz_23_10_2h = read_grd(os.path.join(grd_path, 'fstdbz_202011231000.grd'))[12, 0, :, :]
    dbz_23_12_obs[dbz_23_12_obs<0.0] = 0
    dbz_23_11_1h[dbz_23_11_1h<0.0] = 0
    dbz_23_10_2h[dbz_23_10_2h<0.0] = 0
    dbz_23_dict = {
        'dbz_23_10_2h':dbz_23_10_2h,
        'dbz_23_11_1h':dbz_23_11_1h,
        'dbz_23_12_obs':dbz_23_12_obs,
    }
    rr_21_12_obs = read_grd(os.path.join(grd_path, 'fstdbz_202011211200.grd'))[0, 1, :, :]
    rr_21_11_1h = read_grd(os.path.join(grd_path, 'fstdbz_202011211100.grd'))[6, 1, :, :]
    rr_21_10_2h = read_grd(os.path.join(grd_path, 'fstdbz_202011211000.grd'))[12, 1, :, :]
    rr_21_12_obs[rr_21_12_obs<0.0] = 0
    rr_21_11_1h[rr_21_11_1h<0.0] = 0
    rr_21_10_2h[rr_21_10_2h<0.0] = 0
    rr_21_dict = {
        'rr_21_10_2h':rr_21_10_2h,
        'rr_21_11_1h':rr_21_11_1h,
        'rr_21_12_obs':rr_21_12_obs,
    }
    rr_23_12_obs = read_grd(os.path.join(grd_path, 'fstdbz_202011231200.grd'))[0, 1, :, :]
    rr_23_11_1h = read_grd(os.path.join(grd_path, 'fstdbz_202011231100.grd'))[6, 1, :, :]
    rr_23_10_2h = read_grd(os.path.join(grd_path, 'fstdbz_202011231000.grd'))[12, 1, :, :]
    rr_23_12_obs[rr_23_12_obs<0.0] = 0
    rr_23_11_1h[rr_23_11_1h<0.0] = 0
    rr_23_10_2h[rr_23_10_2h<0.0] = 0
    rr_23_dict = {
        'rr_23_10_2h':rr_23_10_2h,
        'rr_23_11_1h':rr_23_11_1h,
        'rr_23_12_obs':rr_23_12_obs,
    }

    dict_dict = {
        'dbz_21':dbz_21_dict,
        'dbz_23':dbz_23_dict,
        'rr_21':rr_21_dict,
        'rr_23':rr_23_dict,
    }
    for dict_name, dic in dict_dict.items():
        v = [[] for x in range(7)] # store theta(shape: 7*3)
        for data_name, data in dic.items():
            # mpq
            m00=0
            m10=0
            m01=0
            for j in range(y_num) :
                for i in range(x_num) :
                    m00 += data[j,i]*delta*delta
                    m10 += i*data[j,i]*delta*delta
                    m01 += j*data[j,i]*delta*delta
            x_avg = m10/m00
            y_avg = m01/m00

            # cal u
            u00=0
            u20=0
            u02=0
            u30=0
            u03=0
            u12=0
            u21=0
            u11=0
            for j in range(y_num):
                for i in range(x_num):
                    u00 += data[j,i]*delta*delta
                    u20 += ((i-x_avg)**2)*data[j,i]*delta*delta
                    u02 += ((j-y_avg)**2)*data[j,i]*delta*delta
                    u30 += ((i-x_avg)**3)*data[j,i]*delta*delta
                    u03 += ((j-y_avg)**3)*data[j,i]*delta*delta
                    u12 += (i-x_avg)*((j-y_avg)**2)*data[j,i]*delta*delta
                    u21 += ((i-x_avg)**2)*(j-y_avg)*data[j,i]*delta*delta
                    u11 += (i-x_avg)*(j-y_avg)*data[j,i]*delta*delta
            n20=u20/(u00**2)
            n02=u02/(u00**2)
            n11=u11/(u00**2)
            n12=u12/(u00**2.5)
            n21=u21/(u00**2.5)
            n30=u30/(u00**2.5)
            n03=u03/(u00**2.5)

            theta1=n20+n02
            theta2=(n20-n02)**2+4*(n11**2)
            theta3=(n30-3*n12)**2+(3*n21-n03)**2
            theta4=(n30+n12)**2+(n21+n03)**2
            theta5=(n30-3*n12)*(n30+n12)*((n30+n12)**2-3*(n21+n03)**2)+(3*n21-n03)*(n21+n03)*(3*(n30+n12)**2-(n21+n03)**2)
            theta6=(n20-n02)*((n30+n12)**2-(n12+n03)**2)+4*n11*(n30+n12)*(n21+n03)
            theta7=(3*n21-n03)*(n30+n12)*((n30+n12)**2-3*(n21+n03)**2)-(n30-3*n12)*(n21+n03)*(3*(n30+n12)**2-(n21+n03)**2)

            v[0].append(theta1)
            v[1].append(theta2)
            v[2].append(theta3)
            v[3].append(theta4)
            v[4].append(theta5)
            v[5].append(theta6)
            v[6].append(theta7)
        v1_max=max(v[0])
        v1_min=min(v[0])
        v1_scale = v1_max-v1_min
        v2_max=max(v[1])
        v2_min=min(v[1])
        v2_scale = v2_max-v2_min
        v3_max=max(v[2])
        v3_min=min(v[2])
        v3_scale = v3_max-v3_min
        v4_max=max(v[3])
        v4_min=min(v[3])
        v4_scale = v4_max-v4_min
        v5_max=max(v[4])
        v5_min=min(v[4])
        v5_scale = v5_max-v5_min
        v6_max=max(v[5])
        v6_min=min(v[5])
        v6_scale = v6_max-v6_min
        v7_max=max(v[6])
        v7_min=min(v[6])
        v7_scale = v7_max-v7_min

        # 10
        v1_10=(v[0][0]-v1_min)/v1_scale
        v2_10=(v[1][0]-v2_min)/v2_scale
        v3_10=(v[2][0]-v3_min)/v3_scale
        v4_10=(v[3][0]-v4_min)/v4_scale
        v5_10=(v[4][0]-v5_min)/v5_scale
        v6_10=(v[5][0]-v6_min)/v6_scale
        v7_10=(v[6][0]-v7_min)/v7_scale
        # 11
        v1_11=(v[0][1]-v1_min)/v1_scale
        v2_11=(v[1][1]-v2_min)/v2_scale
        v3_11=(v[2][1]-v3_min)/v3_scale
        v4_11=(v[3][1]-v4_min)/v4_scale
        v5_11=(v[4][1]-v5_min)/v5_scale
        v6_11=(v[5][1]-v6_min)/v6_scale
        v7_11=(v[6][1]-v7_min)/v7_scale
        # 12
        v1_12=(v[0][2]-v1_min)/v1_scale
        v2_12=(v[1][2]-v2_min)/v2_scale
        v3_12=(v[2][2]-v3_min)/v3_scale
        v4_12=(v[3][2]-v4_min)/v4_scale
        v5_12=(v[4][2]-v5_min)/v5_scale
        v6_12=(v[5][2]-v6_min)/v6_scale
        v7_12=(v[6][2]-v7_min)/v7_scale

        vd1=[abs(v1_10-v1_12),abs(v1_11-v1_12)]
        vd2=[abs(v2_10-v2_12),abs(v2_11-v2_12)]
        vd3=[abs(v3_10-v3_12),abs(v3_11-v3_12)]
        vd4=[abs(v4_10-v4_12),abs(v4_11-v4_12)]
        vd5=[abs(v5_10-v5_12),abs(v5_11-v5_12)]
        vd6=[abs(v6_10-v6_12),abs(v6_11-v6_12)]
        vd7=[abs(v7_10-v7_12),abs(v7_11-v7_12)]

        #D(pi,p2)
        d_2h=((v1_10-v1_12)**2+(v2_10-v2_12)**2+(v3_10-v3_12)**2+(v4_10-v4_12)**2+(v5_10-v5_12)**2+(v6_10-v6_12)**2+(v7_10-v7_12)**2)**0.5
        d_1h=((v1_11-v1_12)**2+(v2_11-v2_12)**2+(v3_11-v3_12)**2+(v4_11-v4_12)**2+(v5_11-v5_12)**2+(v6_11-v6_12)**2+(v7_11-v7_12)**2)**0.5
        d_max=((max(vd1))**2+(max(vd2))**2+(max(vd3))**2+(max(vd4))**2+(max(vd5))**2+(max(vd6))**2+(max(vd7))**2)**0.5

        #DS=1-(D/D_max)
        d_2hs=1-(d_2h/d_max)
        d_1hs=1-(d_1h/d_max)

        #AS
        as_2h=1-math.acos((v1_10*v1_12+v2_10*v2_12+v3_10*v3_12+v4_10*v4_12+v5_10*v5_12+v6_10*v6_12+v7_10*v7_12)/((v1_10**2+v2_10**2+v3_10**2+v4_10**2+v5_10**2+v6_10**2+v7_10**2)**0.5*(v1_12**2+v2_12**2+v3_12**2+v4_12**2+v5_12**2+v6_12**2+v7_12**2)**0.5))/np.pi
        as_1h=1-math.acos((v1_11*v1_12+v2_11*v2_12+v3_11*v3_12+v4_11*v4_12+v5_11*v5_12+v6_11*v6_12+v7_11*v7_12)/((v1_11**2+v2_11**2+v3_11**2+v4_11**2+v5_11**2+v6_11**2+v7_11**2)**0.5*(v1_12**2+v2_12**2+v3_12**2+v4_12**2+v5_12**2+v6_12**2+v7_12**2)**0.5))/np.pi
        #S=(DS+AS)/2
        s_2h=(d_2hs+as_2h)/2
        s_1h=(d_1hs+as_1h)/2

        print('----> ', dict_name)
        print('s_1h, s_2h:', s_1h, s_2h)
        #print('v_10:', v1_10,v2_10,v3_10,v4_10,v5_10,v6_10,v7_10)
        #print('v_11:', v1_11,v2_11,v3_11,v4_11,v5_11,v6_11,v7_11)
        #print('v_12:', v1_12,v2_12,v3_12,v4_12,v5_12,v6_12,v7_12)
        print('theta7:', v[6][0], v[6][1], v[6][2])



# print dbz verification score and save img (day 21)(10+2 vs 12 / 11+1 vs 12)
def score():
    print('----> score gogo')
    savepath = os.path.join('.', 'data_output_img', 'score')
    # read grd
    dbz_21_12_obs = read_grd(os.path.join(grd_path, 'fstdbz_202011211200.grd'))[0, 0, :, :]
    dbz_21_11_1h = read_grd(os.path.join(grd_path, 'fstdbz_202011211100.grd'))[6, 0, :, :]
    dbz_21_10_2h = read_grd(os.path.join(grd_path, 'fstdbz_202011211000.grd'))[12, 0, :, :]
    dbz_23_12_obs = read_grd(os.path.join(grd_path, 'fstdbz_202011231200.grd'))[0, 0, :, :]
    dbz_23_11_1h = read_grd(os.path.join(grd_path, 'fstdbz_202011231100.grd'))[6, 0, :, :]
    dbz_23_10_2h = read_grd(os.path.join(grd_path, 'fstdbz_202011231000.grd'))[12, 0, :, :]

    threshold = 10.0
    # !!! order is important !!! (< first, > after)
    dbz_21_12_obs[dbz_21_12_obs <= threshold] = 2
    dbz_21_12_obs[dbz_21_12_obs >  threshold] = 1
    dbz_21_11_1h[dbz_21_11_1h <= threshold] = 0.5
    dbz_21_11_1h[dbz_21_11_1h >  threshold] = 1
    dbz_21_10_2h[dbz_21_10_2h <= threshold] = 0.5
    dbz_21_10_2h[dbz_21_10_2h >  threshold] = 1
    index_21_1h = dbz_21_12_obs-dbz_21_11_1h
    index_21_2h = dbz_21_12_obs-dbz_21_10_2h

    dbz_23_12_obs[dbz_23_12_obs <= threshold] = 2
    dbz_23_12_obs[dbz_23_12_obs >  threshold] = 1
    dbz_23_11_1h[dbz_23_11_1h <= threshold] = 0.5
    dbz_23_11_1h[dbz_23_11_1h >  threshold] = 1
    dbz_23_10_2h[dbz_23_10_2h <= threshold] = 0.5
    dbz_23_10_2h[dbz_23_10_2h >  threshold] = 1
    index_23_1h = dbz_23_12_obs-dbz_23_11_1h
    index_23_2h = dbz_23_12_obs-dbz_23_10_2h

    index_field_dict = {
        'index_21_1h':index_21_1h,
        'index_21_2h':index_21_2h,
        'index_23_1h':index_23_1h,
        'index_23_2h':index_23_2h,
    }

    for index_field_name, index_field in index_field_dict.items():
        hit, falal, miss, non = 0, 0, 0, 0
        for j in range(y_num) :
            for i in range(x_num) :
                if index_field[j, i] == 0:
                    hit += 1
                elif index_field[j, i] == 1:
                    falal += 1
                elif index_field[j, i] == 0.5:
                    miss += 1
                elif index_field[j, i] == 1.5:
                    non += 1
                else:
                    print('count error!!!!!!!!!!')
        print('{} hit: {}'.format(index_field_name, str(hit)))
        print('{} falal: {}'.format(index_field_name, str(falal)))
        print('{} miss: {}'.format(index_field_name, str(miss)))
        print('{} non: {}'.format(index_field_name, str(non)))

        plot_basic_contourf('index', index_field, index_field_name, savepath, index_field_name)


if __name__ == '__main__':
    # data_source : ..
    check_output_folder()

    #create_NCDR_maple_img(delete_png_flag=False)
    #compare_forecasts_effectiveness()

    k_mean_convectivecell_marking_v1()
    k_mean_convectivecell_marking_v2()

    #pearson()
    #moment()
    #score()

    # TODO
    # 1.