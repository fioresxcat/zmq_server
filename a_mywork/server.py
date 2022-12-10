import sys
import numpy as np
import cv2
import imagezmq
from sort import *
from my_utils import *
import simplejpeg


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


tracker = Sort()
feature_extractor = None


print('waiting for images')
with imagezmq.ImageHub() as image_hub:
    while True:  # show streamed images until Ctrl-C
        try:
            info, jpg_buffer = image_hub.recv_jpg()
            image_hub.send_reply(b'OK')

            # decode image
            # image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)  # see opencv docs for info on -1 parameter
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

                    if len(boxes) > 0:
                        # track
                        boxes_to_track = np.array(boxes)[:, :-1]
                        tracked_boxes_and_ids = np.array(tracker.update(boxes_to_track))
                        ids = tracked_boxes_and_ids[:, -1]
                        tracked_boxes = tracked_boxes_and_ids[:,:-1].astype(np.int16)

                        # extract features
                        features = []
                        if feature_extractor is not None:
                            for box in tracked_boxes:
                                feature = feature_extractor.extract(image, box)
                                features.append(feature)

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

                        # plot boxes and ids onto image
                        for i, box in enumerate(tracked_boxes):
                            image = cv2.rectangle(image, (int(box[0]), int(
                                box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                            image = cv2.putText(image, str(ids[i]), (int(box[0]), int(
                                box[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.imshow(cam_name, image)  # 1 window for each camera
                cv2.waitKey(1)

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
                            tracker.update(boxes_to_track))
                        ids = tracked_boxes_and_ids[:, -1]
                        tracked_boxes = tracked_boxes_and_ids[:,
                                                              :-1].astype(np.int16)

                        # extract features
                        features = []
                        if feature_extractor is not None:
                            for box in tracked_boxes:
                                feature = feature_extractor.extract(image, box)
                                features.append(feature)

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
                            min_id = check_cosine_similarity(
                                feature, cam1_table)
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

                cv2.imshow(cam_name, image)  # 1 window for each camera
                cv2.waitKey(1)

        except Exception as e:
            print(e)
            continue
