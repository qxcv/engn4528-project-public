"""Detect and align faces."""

import numpy as np
import cv2

from fidentify.utils import data_path, rgb2grey, crop


""" 
The basic structure is based on the github project : https://github.com/roblourens/facealign
All ratio and threshold value refer to paper "New “golden” ratios for facial beauty"

"""
EYEPAIR_RATIO = 2
EYE_MIN_DISTANCE = .35
EYE_MAX_DISTANCE = .5
EYE_MAX_SIZE_DIFFERENCE = 2
EYEPAIR_WIDTH_TO_EYE_WIDTH = .6
FACE_WIDTH_TO_EYE_WIDTH = .4
# Extra pixels to add to input image so that all faces are detected
PAD = 500

# The location of eye on the image
EYE_RATIO_WIDTH = .5
EYE_RATIO_HEIGHT = .3


class FacePipe:
    """FacePipe®: Images Go In, Faces Come Out™"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            data_path('haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(
            data_path('haarcascade_eye.xml'))
        self.eye_pair_cascade = cv2.CascadeClassifier(
            data_path('haarcascade_mcs_eyepair_big.xml'))
        self.left_eye_cascade = cv2.CascadeClassifier(
            data_path('haarcascade_lefteye_2splits.xml'))
        self.right_eye_cascade = cv2.CascadeClassifier(
            data_path('haarcascade_righteye_2splits.xml'))
        self.mouth_cascade = cv2.CascadeClassifier(
            data_path('haarcascade_mcs_mouth.xml'))
        self.nose_cascade = cv2.CascadeClassifier(
            data_path('haarcascade_mcs_nose.xml'))

    def enumerate_faces(self, img, resize_wh=128):
        # Padding the image
        img = cv2.copyMakeBorder(
            img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=255)
        grey = rgb2grey(img)

        # Larger faces has higher priority to solve the problem one person
        # detected as two or more faces
        faces = self.face_cascade.detectMultiScale(grey, 1.3, 5)
        faces = sorted(faces, key=lambda tup: tup[2], reverse=True)

        for face in faces:
            aligned_face = self.face_align(grey, face)
            aligned_face = cv2.bilateralFilter(aligned_face, 0, 5, 2)
            equaliser = cv2.createCLAHE(clipLimit=1.5)
            aligned_face = equaliser.apply(aligned_face)
            resized_aligned_face = cv2.resize(
                aligned_face, (resize_wh, resize_wh),
                interpolation=cv2.INTER_CUBIC)
            # remove the PAD pixels we put in around the main image so
            # coordinates are right
            unpad_face = (face[0] - PAD, face[1] - PAD, face[2], face[3])

            # give back position (for plotting) and cropped image
            yield unpad_face, resized_aligned_face

    def face_align(self, img, face):

        image_width = np.size(img, 1)
        image_height = np.size(img, 0)
        (x, y, w, h) = face

        # Get Target eye width
        # EYEW_TARGET_RATIO = .25
        EYEW_TARGET_RATIO = .25
        EYEW_TARGET = h * EYEW_TARGET_RATIO

        # Get target mouth and eye height
        MOUTH_EYE_TARGET_RATIO = .19
        MOUTH_EYE_TARGET = h * MOUTH_EYE_TARGET_RATIO

        # Get target nose and eye height
        NOSE_EYE_TARGET_RATIO = .12
        NOSE_EYE_TARGET = h * NOSE_EYE_TARGET_RATIO

        grey = rgb2grey(img)
        roi_grey = crop(grey, x, y, w, h)

        eye_pair = self._getEyePair(roi_grey)
        lEye, rEye = self._getEyes(roi_grey, face, eye_pair)
        mouth = self._getMouth(roi_grey)
        nose = self._getNose(roi_grey)

        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)



        # If two eyes are detected

        if lEye is not None and rEye is not None:
            eyeAngle = np.degrees(
                np.arctan((rEye.center[1] - lEye.center[1]) / (rEye.center[
                    0] - lEye.center[0])))
            mid_eye = np.mean([lEye.center, rEye.center], axis=0)
            eye_width = np.linalg.norm(lEye.center - rEye.center)

        # if not two eyes are detected
        else:
            eyeAngle = 0
            if eye_pair is not None:
                mid_eye = eye_pair.center
                eye_width = eye_pair.w * EYEPAIR_WIDTH_TO_EYE_WIDTH
            else:
                mid_eye = np.array([w * EYE_RATIO_WIDTH, h * EYE_RATIO_HEIGHT])
                eye_width = w * FACE_WIDTH_TO_EYE_WIDTH

        # Convert relative coordinate to absolute coordinate
        mid_eye_x = mid_eye[0] + x
        mid_eye_y = mid_eye[1] + y

        mouth_eye_dist = 0

        if mouth is not None:
            mouth_eye_dist = np.linalg.norm(mouth.center - mid_eye)

        nose_eye_dist = 0
        if nose is not None:
            nose_eye_dist = np.linalg.norm(nose.center - mid_eye)

        # Get the maximal width and height based on the eyewidth,
        # mouth_eye_distance and nose_eye_distance
        new_w = max(
            int((eye_width / EYEW_TARGET) * w),
            int((mouth_eye_dist / MOUTH_EYE_TARGET) * w),
            int((nose_eye_dist / NOSE_EYE_TARGET) * w))
        new_h = max(
            int((eye_width / EYEW_TARGET) * h),
            int((mouth_eye_dist / MOUTH_EYE_TARGET) * h),
            int((nose_eye_dist / NOSE_EYE_TARGET) * h))

        # Get homography matrix
        pts_src = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        pts_dst = np.array([[
            mid_eye_x - new_w * EYE_RATIO_WIDTH,
            mid_eye_y - new_h * EYE_RATIO_HEIGHT
        ], [
            mid_eye_x + new_w * (1 - EYE_RATIO_WIDTH),
            mid_eye_y - new_h * EYE_RATIO_HEIGHT
        ], [
            mid_eye_x - new_w * EYE_RATIO_WIDTH,
            mid_eye_y + new_h * (1 - EYE_RATIO_HEIGHT)
        ], [
            mid_eye_x + new_w * (1 - EYE_RATIO_WIDTH),
            mid_eye_y + new_h * (1 - EYE_RATIO_HEIGHT)
        ]])

        homo, status = cv2.findHomography(pts_src, pts_dst)
        img = cv2.warpPerspective(img, homo, (image_width, image_height))

        # Rotation
        if eyeAngle != 0:
            rotMatrix = cv2.getRotationMatrix2D((mid_eye_x, mid_eye_y),
                                                eyeAngle, 1)
            img = cv2.warpAffine(img, rotMatrix, (image_width, image_height))

        return crop(img,
                    int(mid_eye_x - new_w * EYE_RATIO_WIDTH),
                    int(mid_eye_y - new_h * EYE_RATIO_HEIGHT), new_w, new_h)

    def _getNose(self, roi_grey):
        noses = to_rects(self.nose_cascade.detectMultiScale(roi_grey))

        if not noses:
            return None
        else:
            # Find the largest pair
            largest = max(noses, key=lambda e: e.a)

        return largest

    def _getMouth(self, roi_grey):
        mouths = to_rects(self.mouth_cascade.detectMultiScale(roi_grey))
        if not mouths:
            return None
        else:
            # Find the largest pair
            largest = max(mouths, key=lambda e: e.a)

        return largest

    def _getEyePair(self, roi_grey):
        # Convert every eyepairs to Rect Class
        eyepairs = to_rects(self.eye_pair_cascade.detectMultiScale(roi_grey))

        if not eyepairs:
            return None

        # Find the largest eyepair
        largest = max(eyepairs, key=lambda e: e.a)

        if largest.w / largest.h < EYEPAIR_RATIO:
            return None
        else:
            return largest

    def _getEyes(self, roi_grey, face, eyepair):

        # Convert all left eyes and right eyes to Rect Class
        lEyes = to_rects(self.left_eye_cascade.detectMultiScale(roi_grey))
        rEyes = to_rects(self.right_eye_cascade.detectMultiScale(roi_grey))

        if len(lEyes) == 0 or len(rEyes) == 0:
            return (None, None)

        # Filter eye results by having centers in the correct half of the
        # eyepair
        if eyepair:
            rightEyepair, leftEyepair = eyepair.vsplit()

            lEyes = list(
                filter(lambda e: leftEyepair.contains(e.center), lEyes))
            rEyes = list(
                filter(lambda e: rightEyepair.contains(e.center), rEyes))

        if len(lEyes) == 0 or len(rEyes) == 0:
            return (None, None)

        lEye = max(lEyes, key=lambda e: e.a)
        rEye = max(rEyes, key=lambda e: e.a)

        # Throw out the eyes if they are too close
        eyeDist = np.linalg.norm(lEye.center - rEye.center)
        minEyeDist = EYE_MIN_DISTANCE * face[2]
        if eyeDist < minEyeDist:
            return (None, None)

        # Throw out the eyes if they are too far
        maxEyeDist = EYE_MAX_DISTANCE * face[2]
        if eyeDist > maxEyeDist:
            return (None, None)

        # Throw out the eyes if they differ in size too much
        eyeSizeDiff = max(lEye.a, rEye.a) / min(lEye.a, rEye.a)
        if eyeSizeDiff >= EYE_MAX_SIZE_DIFFERENCE:
            return (None, None)

        return (lEye, rEye)


"""
Class structure are copied from https://github.com/roblourens/facealign

"""


class Rect:
    def __init__(self, xywh):
        self.x, self.y, self.w, self.h = xywh
        self.a = self.w * self.h
        self.center = np.array([self.x + self.w / 2.0, self.y + self.h / 2.0])

    def contains(self, p):
        px, py = p
        return self.x <= px <= self.x + self.w and \
            self.y <= py <= self.y + self.h

    def vsplit(self):
        lRect = Rect((self.x, self.y, self.w / 2.0, self.h))
        rRect = Rect((self.center[0], self.y, self.w / 2.0, self.h))
        return lRect, rRect


def to_rects(cvResults):
    return [Rect(result) for result in cvResults]
