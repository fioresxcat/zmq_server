import sys
import numpy as np
import cv2
import imagezmq
from sort import *
from my_utils import *
import simplejpeg
from feature_extractor import FeatureExtractor
from gui.gui import *
import threading
import time

# define global variables
cam1_table, cam2_table = {}, {}
MAX_NUM_FRAME_FROM_LAST_TIME_UPDATE = -1
MAX_NUM_FEATURES = 50
MAX_NUM_FRAME_FROM_LAST_TIME_SEE = 20

MAX_CHECK_OBSOLETE_INTERVAL = MAX_NUM_FRAME_FROM_LAST_TIME_SEE * 2
cam1_to_cam2_counter = {}
cam1_to_cam2_final, cam2_to_cam1_final = {}, {}
global_ls_stay_time = {}
check_obsolete_counter_1, check_obsolete_counter_2 = 0, 0


# util functions
def show_table():
    # print(len(cam2_to_cam1_final.items()))
    cam1_to_cam2 = None
    if len(cam2_to_cam1_final.items()) > 0:
        cam1_to_cam2 = {v:k for k, v in cam2_to_cam1_final.items()}
    # print('cam1 to cam 2: ', cam1_to_cam2)
    print('Cam1_ID\t\tFirst_Time_See\t\tNum_Features\t\tCam2_ID\t\tStay_Time')
    for id, data in cam1_table.items():
        if cam1_to_cam2 is not None and id in cam1_to_cam2.keys():
            # if cam1_to_cam2[id] in global_ls_stay_time.keys():
                # print('\n{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(id, data['first_time_see'], len(data['features']), cam1_to_cam2[id]), global_ls_stay_time[cam1_to_cam2[id]])
            # else:
            print('\n{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(id, data['first_time_see'], len(data['features']), cam1_to_cam2[id], ''))

        else:
            print('\n{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(id, data['first_time_see'], len(data['features']), '', ''))
    
    print('---------------------------------------------------')


tracker_1 = Sort()
tracker_2 = Sort()
feature_extractor = FeatureExtractor()


class Worker(QThread):
    ImageUpdate1 = pyqtSignal(QImage)
    ImageUpdate2 = pyqtSignal(QImage)
    InfoUpdate = pyqtSignal(str)
    def run(self):
        print('waiting for images')
        start_cam1, start_cam2 = 0, 0
        with imagezmq.ImageHub() as image_hub:
            self.ThreadActive = True
            while self.ThreadActive:  # show streamed images until Ctrl-C
                try:
                    info, jpg_buffer = image_hub.recv_jpg()
                    # print('info: ', info)
                    image_hub.send_reply(b'OK')

                    # decode image
                    image = simplejpeg.decode_jpeg(jpg_buffer, colorspace='BGR')

                    if info == 'nano1':
                        cv2.imshow(info, image)  # 1 window for each RPi
                        cv2.waitKey(1)

                    # ---------------------------------------------------- cam 1 ----------------------------------------------------
                    elif info.startswith('cam1'):
                        info = info.split(',')
                        cam_name = info[0]

                        if len(info) > 2:
                            boxes_info = info[1:]

                            # list of [x1, y1, x2, y2, conf_score]
                            boxes = [[int(boxes_info[i]), int(boxes_info[i+1]), int(boxes_info[i+2]), int(
                                boxes_info[i+3]), float(boxes_info[i+4])] for i in range(0, len(boxes_info), 5)]
                            # filter out boxes with low confidence score
                            boxes = [box for box in boxes if box[4] > 0.5]
                            # print('boxes before track: ', boxes)

                            if len(boxes) > 0:
                                # track
                                boxes_to_track = np.array(boxes)[:, :-1]
                                tracked_boxes_and_ids = np.array(tracker_1.update(boxes_to_track))
                                ids = tracked_boxes_and_ids[:, -1]
                                tracked_boxes = tracked_boxes_and_ids[:,:-1].astype(np.int16)

                                # extract features
                                features = np.empty((0, 320))
                                if feature_extractor is not None:
                                    ls_roi = [image[y1:y2, x1:x2] for (x1, y1, x2, y2) in tracked_boxes]
                                    features = feature_extractor.inference(ls_roi)

                                # processing table data
                                current_time = time.time()
                                for feature, id in zip(features, ids):
                                    if id in cam1_table.keys():    # neu item id da co trong table
                                        # get object from table
                                        myobject = cam1_table[id]

                                        # update object with condition
                                        if myobject['num_frame_from_last_time_update'] >= MAX_NUM_FRAME_FROM_LAST_TIME_UPDATE and len(myobject['features']) < MAX_NUM_FEATURES:
                                            myobject['features'].append(feature)
                                            myobject['num_frame_from_last_time_update'] = -1

                                        # if not update
                                        # assign last_time_see (thoi gian gan nhat van nhin thay doi tuong)
                                        myobject['last_time_see'] = current_time
                                        myobject['num_frame_from_last_time_see'] = 0
                                        myobject['num_frame_from_last_time_update'] += 1

                                    elif id != -1:   # neu item id chua co trong table
                                        # them doi tuong vao bang
                                        cam1_table[id] = {
                                            'first_time_see': current_time,
                                            'features': [feature],
                                            'last_time_see': current_time,
                                            'num_frame_from_last_time_see': 0,
                                            'num_frame_from_last_time_update': 0
                                        }
                                        temp = time2datetime(current_time)
                                        print(f'id {cam2_id} comes in at {temp}')
                                        self.InfoUpdate.emit(f'id {cam2_id} comes in at {temp}')

                                # plot boxes and ids onto image
                                for i, box in enumerate(tracked_boxes):
                                    image = cv2.rectangle(image, (int(box[0]), int(
                                        box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                                    image = cv2.putText(image, str(ids[i]), (int(box[0]), int(
                                        box[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        time_elapses = time.perf_counter() - start_cam1
                        start_cam1 = time.perf_counter()
                        fps = 1 / time_elapses
                        image = cv2.putText(image, str(fps), (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # cv2.imshow(cam_name, image)  # 1 window for each camera
                        # cv2.waitKey(1)
                        if cam_name == 'cam1':
                            self.ImageUpdate1.emit(self.convertQt(image))
                        elif cam_name == 'cam2':
                            self.ImageUpdate2.emit(self.convertQt(image))
                        show_table()
                        print('cam2_to_cam1_final: ', cam2_to_cam1_final)


                    # ---------------------------------------------------- cam 2 ----------------------------------------------------
                    elif info.startswith('cam2'):
                        info = info.split(',')
                        cam_name = info[0]

                        if len(info) > 2:
                            boxes_info = info[1:]

                            # list of [x1, y1, x2, y2, conf_score]
                            boxes = [[int(boxes_info[i]), int(boxes_info[i+1]), int(boxes_info[i+2]), int(
                                boxes_info[i+3]), float(boxes_info[i+4])] for i in range(0, len(boxes_info), 5)]
                            # filter out boxes with low confidence score
                            boxes = [box for box in boxes if box[4] > 0.5]

                            if len(boxes) > 0:
                                # track
                                boxes_to_track = np.array(boxes)[:, :-1]
                                tracked_boxes_and_ids = np.array(
                                    tracker_2.update(boxes_to_track))
                                ids = tracked_boxes_and_ids[:, -1]
                                tracked_boxes = tracked_boxes_and_ids[:, :-1].astype(np.int16)

                                # extract features
                                features = np.empty((0, 320))
                                if feature_extractor is not None:
                                    ls_roi = [image[y1:y2, x1:x2] for (x1, y1, x2, y2) in tracked_boxes]
                                    features = feature_extractor.inference(ls_roi)

                                # processing table data
                                current_time = time.time()
                                for feature, id in zip(features, ids):
                                    if id in cam2_table.keys():    # neu item id da co trong table
                                        # get object from table
                                        myobject = cam2_table[id]

                                        # update object with condition
                                        if myobject['num_frame_from_last_time_update'] >= MAX_NUM_FRAME_FROM_LAST_TIME_UPDATE and len(myobject['features']) < MAX_NUM_FEATURES:
                                            myobject['features'].append(feature)
                                            myobject['num_frame_from_last_time_update'] = -1

                                        # if not update
                                        # assign last_time_see (thoi gian gan nhat van nhin thay doi tuong)
                                        myobject['last_time_see'] = current_time
                                        myobject['num_frame_from_last_time_see'] = 0
                                        myobject['num_frame_from_last_time_update'] += 1

                                    elif id != -1:   # neu item id chua co trong table
                                        # them doi tuong vao bang
                                        cam2_table[id] = {
                                            'first_time_see': current_time,
                                            'features': [feature],
                                            'last_time_see': current_time,
                                            'num_frame_from_last_time_see': 0,
                                            'num_frame_from_last_time_update': 0
                                        }

                                    # check feature hien tai cua item nay co giong voi feature nao trong bang cam 1 khong (có thể các lần khác chưa giống nhưng lần này lại giống)
                                    min_id = check_cosine_similarity(feature, cam1_table)
                                    if min_id != -1:  # nếu có
                                        # nếu id này chưa xuất hiện trong cam2_to_cam1_final (lần đầu tiên có cái giống nó ở cam 1), thêm nó vào
                                        if id not in cam2_to_cam1_final:
                                            cam2_to_cam1_final[id] = min_id
                                            # thêm cặp (min_id, id) này vào cam1_to_cam2_counter
                                            cam1_to_cam2_counter[(min_id, id)] = 1

                                        else:
                                            # nếu cặp (min_id, id) này đã có từ trước đó, tăng độ củng cố 2 cái này là một lên 1
                                            if (min_id, id) in cam1_to_cam2_counter:
                                                # (cam1_id, cam2_id)
                                                cam1_to_cam2_counter[(min_id, id)] += 1
                                                # cập nhật id trong cam 1 giống nó nhất
                                                if cam1_to_cam2_counter[(min_id, id)] > cam1_to_cam2_counter[(cam2_to_cam1_final[id], id)]:
                                                    cam2_to_cam1_final[id] = min_id
                                                    # print(f'ID {id} in cam 2 is now matched with ID {min_id} in cam 1')

                                            # nếu cặp (min_id, id) này chưa có (dù id đã có rồi, chứng tỏ lại có 1 thằng min_id khác ở cam 1 giống thằng id này ở cam 2) => thêm vào
                                            else:
                                                cam1_to_cam2_counter[(min_id, id)] = 1

                                # plot boxes and ids onto image
                                for i, box in enumerate(tracked_boxes):
                                    image = cv2.rectangle(image, (int(box[0]), int(
                                        box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                                    image = cv2.putText(image, str(cam2_to_cam1_final[ids[i]]), (int(
                                        box[0]), int(box[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                
                            # duyet qua tat ca nhung id khong xuat hien o lan nhan nay trong bang cam 2
                            ls_id_received = ids if len(boxes) > 0 else []
                            ls_delete = []
                            for cam2_id, cam2_object in cam2_table.items():
                                # nếu id ko trong đống id vừa nhận được
                                if cam2_id not in ls_id_received:
                                    # tăng số frame từ lần nhìn thấy gần nhất
                                    cam2_object['num_frame_from_last_time_see'] += 1
                                    cam2_object['num_frame_from_last_time_update'] += 1

                                    # nếu số frame từ lần nhìn thấy gần nhất lớn hơn ngưỡng cho phép => xóa đối tượng này
                                    if cam2_object['num_frame_from_last_time_see'] >= MAX_NUM_FRAME_FROM_LAST_TIME_SEE:
                                        # get doi tuong tuong ung trong bang cam 1
                                        if cam2_id in cam2_to_cam1_final:
                                            correspond_cam1_id = cam2_to_cam1_final[cam2_id]
                                            correspond_cam1_object = cam1_table[correspond_cam1_id]
                                            # nếu số frame từ lần nhìn thấy gần nhất ở cam 1 cũng lớn hơn ngưỡng cho phép => xóa đối tượng này
                                            # if correspond_cam1_object['num_frame_from_last_time_see'] >= MAX_NUM_FRAME_FROM_LAST_TIME_SEE:
                                            if True:
                                                come_out_time = max(cam2_object['last_time_see'], correspond_cam1_object['last_time_see'])
                                                come_in_time = min(cam2_object['first_time_see'], correspond_cam1_object['first_time_see'])
                                                stay_time = come_out_time - come_in_time
                                                global_ls_stay_time[correspond_cam1_id] = stay_time
                                                ls_delete.append(cam2_id)
                                            print('ID {} comes out at {}. Stay time: {}'.format(correspond_cam1_id, come_out_time, stay_time))
                                            self.InfoUpdate.emit('ID {} comes out at {}. Stay time: {}'.format(correspond_cam1_id, come_out_time, stay_time))

                                        else:  # nếu ko có đối tượng tương ứng trong bảng cam 1
                                            come_out_time = cam2_object['last_time_see']
                                            come_in_time = cam2_object['first_time_see']
                                            stay_time = come_out_time - come_in_time
                                            global_ls_stay_time[correspond_cam1_id] = stay_time
                                            ls_delete.append(cam2_id)

                            
                            # xoa
                            for id in ls_delete:
                                # xoa object tuong ung trong bang cam 1
                                try:
                                    del cam1_table[cam2_to_cam1_final[id]]
                                except:
                                    pass
                                
                                # xoa object trong cam 2
                                del cam2_table[id]

                                # xoa object tuong ung trong cam1_to_cam2_counter
                                try:
                                    ls_keys = list(cam1_to_cam2_counter.keys())
                                    for key in ls_keys:
                                        if key[0] == cam2_to_cam1_final[id]:
                                            del cam1_to_cam2_counter[key]
                                except:
                                    pass

                                try:
                                    del cam2_to_cam1_final[id]
                                except:
                                    pass
                                
                        time_elapses = time.perf_counter() - start_cam2
                        start_cam2 = time.perf_counter()
                        fps = 1 / time_elapses
                        image = cv2.putText(image, str(fps), (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        # cv2.imshow(cam_name, image)  # 1 window for each camera
                        # cv2.waitKey(1)
                        if cam_name == 'cam1':
                            self.ImageUpdate1.emit(self.convertQt(image))
                        elif cam_name == 'cam2':
                            self.ImageUpdate2.emit(self.convertQt(image))
                        # print('box form cam2: ', tracked_boxes)
                        show_table()
                        print('cam2_to_cam1_final: ', cam2_to_cam1_final)

                except Exception as e:
                    print(e)
                    continue
    
    def convertQt(self, image):
        Image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # FlippedImage = cv2.flip(Image, 1)
        FlippedImage = Image
        ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(960, 720, Qt.KeepAspectRatio)
        return Pic
    
    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    Wk = Worker()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow, Wk)
    MainWindow.show()
    sys.exit(app.exec_())