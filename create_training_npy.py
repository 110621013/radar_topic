import os
import numpy as np

#時間上不連續要怎麼處理？
#--> 實際上有gap的點有2020的7~8月跟2021的2~3月 <--做資料避開他們
#時間序列創建需要考慮要不要重複取到序列？
#--> 先看總資料量再決定
#轉成怎樣形狀的array？
#--> 先看convLSTM的輸入形狀

x_num, y_num, t_num, v_num = 921, 881, 19, 2

#arg: path+filename, return nparray(t, var, y, x)
def read_grd(filename):
    data = np.fromfile(filename, dtype='<f4')
    data = data.reshape((t_num, v_num, y_num, x_num), order='C')
    return data

if __name__ == '__main__':
    dbz_data_x = np.zeros((y_num, x_num)).reshape(1, y_num, x_num)
    rr_data_x = np.zeros((y_num, x_num)).reshape(1, y_num, x_num)
    train_mouth_list = ['202006', '202007', '202008', '202009', '202010', '202011', '202012']

    data_root_path = os.path.join('.', 'data')
    mouth_list = [x for x in os.listdir(data_root_path) if x in train_mouth_list]
    print('->', mouth_list)
    for mouth in mouth_list:
        root_mouth_path = os.path.join(data_root_path, mouth)
        day_list = [x for x in os.listdir(root_mouth_path)]
        print('-->', day_list)
        for day in day_list:
            root_mouth_day_path = os.path.join(root_mouth_path, day)

            # unzip gz file -> grd file
            gz_list = [x for x in os.listdir(root_mouth_day_path) if x.split('.')[-1] =='gz']
            print('--->', gz_list)
            for gz in gz_list:
                os.system('gzip -d {}'.format( os.path.join(root_mouth_day_path, gz) ))

            # get numpy array
            grd_list = [x for x in os.listdir(root_mouth_day_path) if x.split('.')[-1] =='grd']
            for grd_name in grd_list:
                grd_abs_filename = os.path.join(root_mouth_day_path, grd_name)
                tvyx_arr = read_grd(grd_abs_filename)

                dbz_obs, rr_obs = tvyx_arr[0, 0, :, :], tvyx_arr[0, 1, :, :]
                dbz_obs[dbz_obs <= 0], rr_obs[rr_obs <= 0] = np.nan, np.nan

                dbz_data_x = np.vstack((dbz_data_x, dbz_obs))
                rr_data_x = np.vstack((rr_data_x, rr_obs))


            break
        break

    print('-------over------')