import os

if __name__ == '__main__':
    data_root_path = os.path.join('..', 'data')
    mouth_list = [x for x in os.listdir(data_root_path) if x != 'sample']
    print('---> mouth_list:', mouth_list)
    for mouth in mouth_list:
        root_mouth_path = os.path.join(data_root_path, mouth)
        day_list = [x for x in os.listdir(root_mouth_path)]
        print('---> day_list:', day_list)
        for day in day_list:
            root_mouth_day_path = os.path.join(root_mouth_path, day)

            # unzip gz file -> grd file
            gz_list = [x for x in os.listdir(root_mouth_day_path) if x.split('.')[-1] =='gz']
            print('---> gz_list:', gz_list)
            for gz in gz_list:
                os.system('gzip -d {}'.format( os.path.join(root_mouth_day_path, gz) ))

    print('-------over------')