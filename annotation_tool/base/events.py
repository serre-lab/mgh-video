import cv2

from .shapes import Rectangle, Circle
from .annotations import Annotation

class EventHandler(object):
    def trigger(self, ttype):
        return getattr(self, ttype)

    def pointInCircle(self, x, y, rx, ry, R):
        if((x - rx) ** 2) + ((y - ry) ** 2) < R ** 2:
            return True
        else:
            return False
        
    def updateAnnots(self, annotObj, frame_n, image):
        joints = list(annotObj.joints.keys())
        annot_df = annotObj.joints_df[annotObj.joints_df.frame_n == frame_n][joints]
        if annot_df.empty:
            return
        
        annotObj.image = image
        annotObj.frame_n = frame_n
        for joint in annot_df:
            vals = annot_df[joint].values[0].split('-')
            annotObj.joints[joint].x_center, annotObj.joints[joint].y_center = vals[0], vals[1]
            #print(annot_df[joint].values[0].split('-'))

        self.clear_canvas_draw(annotObj)

    def clear_canvas_draw(self, annotObj):
        temp = annotObj.image.copy()
        temp_copy = annotObj.image.copy()

        for joint_name in annotObj.joints:
            joint = annotObj.joints[joint_name]
            if joint.x_center == 0:
                return

            x, y, r = int(float(joint.x_center)), int(float(joint.y_center)), int(float(joint.radius))
            cv2.circle(temp, (x, y), r, annotObj.colorDict[joint_name], -1)
            if joint.focus:
                cv2.circle(temp, (x, y), r, (255, 255, 255), 2)

        # Apply the overlay
        colorList = [[0, 0, 255], [0, 255, 0], [0, 255, 255]]
        qual = annotObj.joints_df['quality'][annotObj.frame_n]
        cv2.circle(temp, (10, 10), 10, colorList[qual], -1)
        cv2.addWeighted(temp, 0.5, temp_copy, 0.5, 0, temp_copy)
        cv2.imshow(annotObj.wname, temp_copy)

    def disableResizeButtons(self, annotObj):
        annotObj.hold = False
    
    def releaseMouseButton(self, x, y, annotObj):
        if annotObj.selectedJoint:
            annotObj.selectedJoint.drag = False
            self.disableResizeButtons(annotObj)
            annotObj.selectedJoint.hold = False
            annotObj.selectedJoint.active = False
            annotObj.selectedJoint = None

            self.clear_canvas_draw(annotObj)

    def pressMouseButton(self, x, y, annotObj):
        if annotObj.selectedJoint:
            return
        else:
            for joint_name in annotObj.joints:
                joint = annotObj.joints[joint_name]
                if joint.x_center == 0:
                    continue

                if self.pointInCircle(x, y, int(float(joint.x_center)), int(float(joint.y_center)), int(float(joint.radius))):
                    annotObj.selectedJoint = annotObj.joints[joint_name]
                    annotObj.selectedJoint.x_center = x
                    annotObj.selectedJoint.y_center = y
                    annotObj.selectedJoint.drag = True
                    annotObj.selectedJoint.active = True
                    annotObj.selectedJoint.hold = True
    
    def moveMousePointer(self, x, y, annotObj):
        if annotObj.selectedJoint:
            joint = annotObj.selectedJoint
            joint.x_center = x
            joint.y_center = y

            if joint.x_center < annotObj.keepWithin.x:
                joint.x_center = annotObj.keepWithin.x
            if joint.y_center < annotObj.keepWithin.y:
                joint.y_center = annotObj.keepWithin.y

            if (joint.x_center + joint.radius) > (annotObj.keepWithin.x + annotObj.keepWithin.width - 1):
                joint.x_center = annotObj.keepWithin.x + annotObj.keepWithin.width - 1 - joint.radius
            if (joint.y_center + joint.radius) > (annotObj.keepWithin.y + annotObj.keepWithin.height - 1):
                joint.y_center = annotObj.keepWithin.y + annotObj.keepWithin.height - 1 - joint.radius

            if annotObj.multiframe:
                annotObj.joints_df.loc[annotObj.joints_df['frame_n'] >= annotObj.frame_n, joint.label] = str(joint.x_center) + '-' + str(joint.y_center) + '-10'
            else:
                annotObj.joints_df.loc[annotObj.joints_df['frame_n'] == annotObj.frame_n, joint.label] = str(joint.x_center) + '-' + str(joint.y_center) + '-10'

            self.clear_canvas_draw(annotObj)

    def mouseDoubleClick(self, x, y, annotObj):
        for joint_name in annotObj.joints:
            joint = annotObj.joints[joint_name]

            if joint.x_center == 0:
                return
            
            if self.pointInCircle(x, y, int(float(joint.x_center)), int(float(joint.y_center)), int(float(joint.radius))):
                joint.focus = not joint.focus
                print(joint_name + ' - Focus ' + str(joint.focus))
            else:
                joint.focus = False

    def occludedJoint(self, annotObj):
        for joint_name in annotObj.joints:
            joint = annotObj.joints[joint_name]
            if joint.focus:
                _, _, score = annotObj.joints_df.loc[annotObj.joints_df['frame_n'] == annotObj.frame_n, joint_name].values[0].split('-')
                if score == '0':
                    annotObj.joints_df.loc[annotObj.joints_df['frame_n'] == annotObj.frame_n, joint_name] = str(joint.x_center) + '-' + str(joint.y_center) + '-10'
                else:
                    annotObj.joints_df.loc[annotObj.joints_df['frame_n'] == annotObj.frame_n, joint_name] = str(joint.x_center) + '-' + str(joint.y_center) + '-0'

                return joint_name
        print("No marker is on focus!")
        return None

    def keyboardMoveMarker(self, x, y, annotObj):
        for joint_name in annotObj.joints:
            joint = annotObj.joints[joint_name]
            if joint.focus:
                annotObj.selectedJoint = joint

        if annotObj.selectedJoint:
            joint = annotObj.selectedJoint
            joint.x_center = x
            joint.y_center = y

            if joint.x_center < annotObj.keepWithin.x:
                joint.x_center = annotObj.keepWithin.x
            if joint.y_center < annotObj.keepWithin.y:
                joint.y_center = annotObj.keepWithin.y

            if (joint.x_center + joint.radius) > (annotObj.keepWithin.x + annotObj.keepWithin.width - 1):
                joint.x_center = annotObj.keepWithin.x + annotObj.keepWithin.width - 1 - joint.radius
            if (joint.y_center + joint.radius) > (annotObj.keepWithin.y + annotObj.keepWithin.height - 1):
                joint.y_center = annotObj.keepWithin.y + annotObj.keepWithin.height - 1 - joint.radius

            if annotObj.multiframe:
                annotObj.joints_df.loc[annotObj.joints_df['frame_n'] >= annotObj.frame_n, joint.label] = str(joint.x_center) + '-' + str(joint.y_center) + '-10'
            else:
                annotObj.joints_df.loc[annotObj.joints_df['frame_n'] == annotObj.frame_n, joint.label] = str(joint.x_center) + '-' + str(joint.y_center) + '-10'

            self.clear_canvas_draw(annotObj)
