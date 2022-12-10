# Code test get facial component
# img =  gt[0][0].permute(1,2,0).cpu().numpy()*255
# img =  np.ascontiguousarray(img, dtype=np.uint8)
# for i in facial_components:
#     x1 = i[0]
#     y1 = i[1]
#     x2 = i[2]
#     y2 = i[3]
#     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
# cv2.imwrite("/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/test_crop_facial_componen/img.jpg",img)

#     left_eyes_gt = left_eyes_gt.permute(0,2,3,1)
# for i in range(6):
#     img = left_eyes_gt[i,:,:,:].cpu().numpy()*255
#     img =  np.ascontiguousarray(img, dtype=np.uint8)
#     cv2.imwrite("/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/test_crop_facial_componen/left_eye_{}.jpg".format(i), img)

# right_eyes_gt = right_eyes_gt.permute(0,2,3,1)
# for i in range(6):
#     img = right_eyes_gt[i,:,:].cpu().numpy()*255
#     img =  np.ascontiguousarray(img, dtype=np.uint8)
#     cv2.imwrite("/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/test_crop_facial_componen/right_eye_{}.jpg".format(i),img)

# mouths_gt = mouths_gt.permute(0,2,3,1)
# for i in range(6):
#     img = mouths_gt[i,:,:].cpu().numpy()*255
#     img =  np.ascontiguousarray(img, dtype=np.uint8)
#     cv2.imwrite("/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/test_crop_facial_componen/mouth_{}.jpg".format(i),img)
