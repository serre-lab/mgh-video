import numpy as np

from annotate_gui import AnnotationGUI

if __name__ == '__main__':
    data_dir = '/media/data_cifs/lakshmi/MGH/mgh-pose/bootstrap_R1_DLC/Kalpit'
    output_dir = '/media/data_cifs/lakshmi/MGH/mgh-pose/bootstrap_R1_DLC/Kalpit'
    gui = AnnotationGUI(data_dir=data_dir, output_dir=output_dir,
            data_ext='avi', with_annots=True,
            annots_file_ext='csv', yaml_config='mgh.yaml')

    root = gui.build_gui()
    root.mainloop()
