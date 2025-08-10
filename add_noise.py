import cv2
import numpy as np
import random

def add_gaussian_noise(image, mean=0, sigma=25):
    """添加高斯噪声到图像"""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return np.clip(noisy_image, 0, 255)

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """添加椒盐噪声到图像"""
    noisy_image = np.copy(image)
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0
    return noisy_image

def add_motion_blur(image, kernel_size=15):
    """添加运动模糊效果"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

def add_low_light_noise(image, darken_factor=0.5, noise_factor=30):
    """模拟低光条件下的噪声"""
    dark_image = (image * darken_factor).astype(np.uint8)
    return add_gaussian_noise(dark_image, sigma=noise_factor)

# 主函数
if __name__ == "__main__":
    input_video = "sample.mp4"
    output_video = "noise.mp4"
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("无法打开视频文件")
        exit()
    
    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        noisy_frame = add_gaussian_noise(frame, sigma=0.75)
        prob = 0.00075
        noisy_frame = add_salt_pepper_noise(noisy_frame, salt_prob=prob, pepper_prob=prob)
        noisy_frame = add_motion_blur(noisy_frame, kernel_size=20)

        # 写入输出视频
        out.write(noisy_frame)
        
        # 显示结果
        cv2.imshow("Noisy Video", noisy_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()