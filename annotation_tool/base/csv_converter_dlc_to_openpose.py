import os, glob

def convert(csvpath):
    with open(csvpath, 'r') as f:
        lines = f.readlines()[4:]

    with open(csvpath.split('.')[0] + '_openpose.csv', 'w') as f:
        header = 'sub_n,video_name,frame_n,nose,r_shoulder,r_elbow,r_wrist,l_shoulder,l_elbow,l_wrist,quality'
        vid_name = os.path.basename(csvpath).split('.')[0]
        sub_n = 0
        f.write(header)
        for line in lines:
            vals = line.strip().split(',')
            strtowrite = "{},{},{},{},{},{},{},{},{},{},{}".format(
                str(sub_n), vid_name, int(vals[0])-1,
                '-'.join(vals[1:4]),
                '-'.join(vals[4:7]),
                '-'.join(vals[7:10]),
                '-'.join(vals[10:13]),
                '-'.join(vals[13:16]),
                '-'.join(vals[16:19]),
                '-'.join(vals[19:22]),
                1
            )
            f.write("\n" + strtowrite)
    return
