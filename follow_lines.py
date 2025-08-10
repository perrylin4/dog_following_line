import cv2
import numpy as np
import time
import random

class LineFollower:
    def __init__(self):
        self.lines_list = []
        self.fx = 656.58771575
        self.fy = 656.60110198
        self.cx = 631.58766775
        self.cy = 527.02964399
        self.edges = None
        self.eroded = None
        self.dilated = None
        self.mask = None
        self.gray = None
        self.lab = None
        self.binary_image = None
        self.line_image = None
        self.debug_image = None
        self.transparent_overlay = None
    
    def create_binary_image(self, image):
        """创建二值化图像"""
        # 1. 使用更高效的LAB转换
        self.lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 2. 优化掩码提取
        lower_black = np.array([0, 60, 60], dtype=np.uint8)
        upper_black = np.array([120, 160, 160], dtype=np.uint8)
        self.mask = cv2.inRange(self.lab, lower_black, upper_black)

        # 3. 优化形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.eroded = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        
        # 4. 存储二值化图像
        self.binary_image = self.eroded.copy()
        
        # 5. 按需创建调试图像
        if self.debug_image is None:
            self.debug_image = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
            self.transparent_overlay = np.zeros_like(self.debug_image, dtype=np.uint8)
        return self.binary_image
    
    def linear_regression(self, points):
        """实现简单线性回归"""
        if len(points) < 2:
            return None, None
            
        # 提取x和y坐标
        x = np.array([p[0] for p in points], dtype=np.float32)
        y = np.array([p[1] for p in points], dtype=np.float32)
        
        # 计算斜率和截距
        n = len(points)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-5)
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
    
    def draw_line_with_alpha(self, slope, intercept, color, width, height, alpha=0.1):
        """绘制具有指定透明度的直线"""
        # 创建临时透明图层
        temp_overlay = np.zeros_like(self.transparent_overlay, dtype=np.uint8)
        
        # 获取直线点
        points = self.get_line_points(slope, intercept, width, height)
        if points:
            point1, point2 = points
            x1, y1 = point1
            x2, y2 = point2
            
            # 在临时图层上绘制直线
            cv2.line(temp_overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 将临时图层以指定透明度混合到主透明图层
            self.transparent_overlay = cv2.addWeighted(
                self.transparent_overlay, 1, 
                temp_overlay, alpha, 
                0
            )
    
    def validate_line(self, slope, intercept, binary_image, num_test_points=50, threshold=0.95):
        """改进的直线验证函数 - 只考虑图像内的点"""
        height, width = binary_image.shape
        valid_count = 0
        total_points_in_image = 0  # 实际在图像内的点数量
        
        # 在图像宽度范围内生成测试点
        x_test = np.linspace(0, width-1, num_test_points)
        y_test = slope * x_test + intercept
        
        # 检查每个点是否在图像范围内且在白色区域内
        for i in range(len(x_test)):
            x, y = x_test[i], y_test[i]
            # 检查点是否在图像内
            if 0 <= y < height and 0 <= x < width:
                total_points_in_image += 1
                if binary_image[int(y), int(x)] > 0:
                    valid_count += 1
        
        # 避免除以零错误
        if total_points_in_image == 0:
            return False, 0.0
        
        # 计算有效点比例（只考虑图像内的点）
        valid_ratio = valid_count / total_points_in_image
        return valid_ratio >= threshold, valid_ratio
    
    def get_line_points(self, slope, intercept, width, height):
        """获取直线在图像边界上的两个点"""
        # 计算直线与图像边界的交点
        points = []
        
        # 与上边界(y=0)的交点
        if abs(slope) > 1e-6:
            x_top = (0 - intercept) / slope
            if 0 <= x_top < width:
                points.append((x_top, 0))
        
        # 与下边界(y=height-1)的交点
        if abs(slope) > 1e-6:
            x_bottom = (height-1 - intercept) / slope
            if 0 <= x_bottom < width:
                points.append((x_bottom, height-1))
        
        # 与左边界(x=0)的交点
        y_left = intercept
        if 0 <= y_left < height:
            points.append((0, y_left))
        
        # 与右边界(x=width-1)的交点
        y_right = slope * (width-1) + intercept
        if 0 <= y_right < height:
            points.append((width-1, y_right))
        
        # 如果找到至少两个点，返回最远的两个点
        if len(points) >= 2:
            max_dist = 0
            best_pair = None
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    dist = np.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (points[i], points[j])
            return best_pair
        
        return None
    
    def draw_line_on_overlay(self, slope, intercept, color, width, height):
        """在透明图层上绘制直线"""
        points = self.get_line_points(slope, intercept, width, height)
        if points:
            point1, point2 = points
            x1, y1 = point1
            x2, y2 = point2
            cv2.line(self.transparent_overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    def strict_avoid_white_line(self, slope, intercept, binary_image, threshold=0.01):
        """严格验证分割线是否避开白线"""
        height, width = binary_image.shape
        
        # 1. 使用高密度采样点
        num_test_points = 100  # 增加采样点数量
        x_test = np.linspace(0, width-1, num_test_points)
        y_test = slope * x_test + intercept
        
        # 2. 检查每个点是否在图像内且在白色区域内
        white_count = 0
        total_in_image = 0
        
        for i in range(len(x_test)):
            x, y = x_test[i], y_test[i]
            if 0 <= y < height and 0 <= x < width:
                total_in_image += 1
                if binary_image[int(y), int(x)] > 0:
                    white_count += 1
        
        # 3. 如果图像内点太少，返回False
        if total_in_image < 20:
            return False
        
        # 4. 计算白点比例
        white_ratio = white_count / total_in_image
        
        # 5. 严格条件：白点比例必须低于阈值
        return white_ratio < threshold
    
    def generate_split_line(self, binary_image, max_attempts=200):
        """优化分割线生成"""
        height, width = binary_image.shape
        
        # 1. 优化随机采样
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            
            # 使用更高效的随机斜率生成
            antislope = random.uniform(-2, 2)  # 限制斜率范围
            if antislope == 0:
                continue
            slope = 1 / antislope
            intercept = random.uniform(0, height) - slope * random.uniform(0, width)
            
            # 快速预检查
            if not self.quick_line_check(slope, intercept, binary_image):
                continue
                
            # 严格验证
            if self.strict_avoid_white_line(slope, intercept, binary_image):
                # 检查分割线是否将图像分为两个区域
                region1, region2 = self.split_image_by_line(slope, intercept, binary_image)
                white_pixels1 = np.sum(region1 > 0)
                white_pixels2 = np.sum(region2 > 0)
                total_white = white_pixels1 + white_pixels2
                
                if total_white > 0 and white_pixels1/total_white > 0.05 and white_pixels2/total_white > 0.05:
                    cv2.line(self.line_image, (0, int(intercept)), (width, int(slope * width + intercept)), (255, 0, 0), 2)
                    return slope, intercept
        # 2. 如果找不到合适的分割线，返回垂直中线
        print("未找到合适的分割线，使用垂直中线")
        cv2.line(self.line_image, (width//2, 0), (width//2, height), (255, 0, 0), 2)
        return 0, width // 2  # 垂直中线，斜率为0，截距为图像宽度的一半

    def quick_line_check(self, slope, intercept, binary_image):
        """快速检查分割线是否可能有效"""
        height, width = binary_image.shape
        
        # 检查4个关键点（左上、右上、左下、右下）
        points = [
            (0, 0),
            (width-1, 0),
            (0, height-1),
            (width-1, height-1)
        ]
        
        white_count = 0
        for x, y in points:
            # 计算点在直线的哪一侧
            if slope * x - y + intercept >= 0:
                if binary_image[y, x] > 0:
                    white_count += 1
        
        # 如果所有关键点都在白线上，可能不是好分割线
        return white_count < 4
    
    def split_image_by_line(self, slope, intercept, binary_image):
        """进一步优化图像分割"""
        height, width = binary_image.shape
        
        # 使用广播机制避免创建大型网格
        y_indices = np.arange(height)[:, np.newaxis]  # 形状: (height, 1)
        x_indices = np.arange(width)                  # 形状: (width,)
        
        # 计算距离 (使用广播)
        distances = slope * x_indices - y_indices + intercept
        
        # 根据距离分割图像
        region1 = np.where(distances >= 0, binary_image, 0)
        region2 = np.where(distances < 0, binary_image, 0)
        
        return region1, region2
    
    def detect_line_in_region(self, region, num_lines=200, max_attempts=1000):
        """优化区域直线检测"""
        # 1. 获取白色像素坐标 (y, x格式)
        white_coords = np.argwhere(region > 0)
        
        # 2. 如果没有足够的点，返回空列表
        if len(white_coords) < 10:
            return []
        
        valid_lines = []
        attempts = 0
        
        # 3. 预计算采样索引
        total_points = len(white_coords)
        sample_indices = np.random.choice(total_points, size=(max_attempts, 10), replace=True)
        
        while len(valid_lines) < num_lines and attempts < max_attempts:
            # 获取采样点 (y, x格式)
            indices = sample_indices[attempts]
            sampled_points = white_coords[indices]
            
            # 拟合直线 (需要转换为(x, y)格式)
            points_xy = [(x, y) for y, x in sampled_points]
            slope, intercept = self.linear_regression(points_xy)
            
            # 如果无法拟合，继续尝试
            if slope is None:
                attempts += 1
                continue
                
            # 验证直线有效性
            is_valid, valid_ratio = self.validate_line(slope, intercept, region)
            # height, width = region.shape[:2]

            if is_valid:
                valid_lines.append((slope, intercept, valid_ratio))
            #     self.draw_line_with_alpha(slope, intercept, (0, 255, 0), width, height, alpha=0.05)
            # else:
            #     self.draw_line_with_alpha(slope, intercept, (0, 0, 255), width, height, alpha=0.2)

            attempts += 1
        return valid_lines
    
    def group_similar_lines(self, lines, angle_threshold=0.1, dist_threshold=100):
        """优化直线聚类"""
        if not lines:
            return []
            
        # 1. 使用更高效的分组方法
        groups = []
        height, width = self.binary_image.shape
        mid_y = height // 2
        
        for slope, intercept, valid_ratio in lines:
            # 计算直线在图像中部的x坐标
            mid_x = (mid_y - intercept) / slope if abs(slope) > 1e-5 else width/2
            
            # 查找匹配的组
            found = False
            for group in groups:
                # 检查角度相似性
                angle_diff = abs(np.arctan(slope) - np.arctan(group[0][0]))
                if angle_diff > angle_threshold:
                    continue
                    
                # 检查位置相似性
                group_mid_x = np.mean([(mid_y - i[1]) / i[0] for i in group if abs(i[0]) > 1e-5])
                if abs(mid_x - group_mid_x) > dist_threshold:
                    continue
                    
                # 匹配成功，加入组
                group.append((slope, intercept, valid_ratio))
                found = True
                break
            
            # 没有找到匹配组，创建新组
            if not found:
                groups.append([(slope, intercept, valid_ratio)])
        
        # 2. 合并每组内的直线
        merged_lines = []
        for group in groups:
            slopes = np.array([slope for slope, _, _ in group])
            intercepts = np.array([intercept for _, intercept, _ in group])
            valid_ratios = np.array([valid_ratio for _, _, valid_ratio in group])
            
            # 计算加权平均值（根据有效性比例）
            weights = valid_ratios / np.sum(valid_ratios)
            avg_slope = np.sum(slopes * weights)
            avg_intercept = np.sum(intercepts * weights)
            avg_valid_ratio = np.mean(valid_ratios)
            
            merged_lines.append((avg_slope, avg_intercept, avg_valid_ratio))
        
        # 3. 按有效性排序
        merged_lines.sort(key=lambda x: x[2], reverse=True)
        return merged_lines
    
    def merge_lines(self, lines):
        """合并多条直线为一条直线（平均斜率和截距）"""
        if not lines:
            return None, None, 0.0
        
        slopes = []
        intercepts = []
        valid_ratios = []
        
        for slope, intercept, valid_ratio in lines:
            slopes.append(slope)
            intercepts.append(intercept)
            valid_ratios.append(valid_ratio)
        
        # 计算加权平均值（根据有效性比例）
        weights = np.array(valid_ratios) / np.sum(valid_ratios)
        avg_slope = np.sum(np.array(slopes) * weights)
        avg_intercept = np.sum(np.array(intercepts) * weights)
        avg_valid_ratio = np.mean(valid_ratios)
        
        return (avg_slope, avg_intercept, avg_valid_ratio)
    
    def follow_lines(self, image):
        self.lines_list = []        
        height, width = image.shape[:2]
        self.line_image = image.copy()
        
        # 添加整体计时
        # total_start = time.time()
        
        # 1. 创建二值化图像
        binary_image = self.create_binary_image(image)
        
        # 2. 生成分割线
        split_slope, split_intercept = self.generate_split_line(binary_image)
        
        # 3. 分割图像为两个区域
        region1, region2 = self.split_image_by_line(split_slope, split_intercept, binary_image)
        
        # 4. 在每个区域内检测直线
        lines_region1 = self.detect_line_in_region(region1)
        lines_region2 = self.detect_line_in_region(region2)
        
        # 5. 聚类和合并直线
        merged_lines1 = self.merge_lines(lines_region1)
        merged_lines2 = self.merge_lines(lines_region2)

        # 6. 选择最优的两条直线（每个区域一条）
        selected_lines = []
        if merged_lines1 and merged_lines1[0] and merged_lines1[1]:
            selected_lines.append(merged_lines1)
        if merged_lines2 and merged_lines2[0] and merged_lines2[1]:
            selected_lines.append(merged_lines2)

        # self.line_image = cv2.addWeighted(self.line_image, 1, self.transparent_overlay, 1, 0)
        
        # 如果某个区域没有检测到直线，尝试全局检测
        if len(selected_lines) < 2:
            return 0.0  # 没有足够的直线进行跟踪

        # 7. 绘制直线和计算中心
        center_x = 0
        valid_lines_count = 0

        for i, (slope, intercept, valid_ratio) in enumerate(selected_lines):
            # 获取直线在图像边界上的两个点
            if not slope:
                raise Exception("未检测到有效直线")
            points = self.get_line_points(slope, intercept, width, height)
            
            if points:
                point1, point2 = points
                x1, y1 = point1
                x2, y2 = point2
                
                # 计算线段中点
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                center_x += mid_x
                valid_lines_count += 1

                if True:
                    # 绘制分割线
                    self.draw_line_on_overlay(slope, intercept, (0, 255, 255), width, height)

                    # 绘制线段
                    color = (0, 0, 255)  # 红色
                    cv2.line(self.line_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

                    # 绘制线段中点
                    cv2.circle(self.line_image, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)
                    
                    # 显示有效性比例
                    cv2.putText(self.line_image, f"Valid: {valid_ratio:.2f}", 
                            (int(mid_x)+10, int(mid_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if valid_lines_count > 0:
            center_x /= valid_lines_count
            # print(f"检测到 {valid_lines_count} 条有效线段")
        else:
            return 0.0  # 没有有效线段，无法计算误差

        # total_time = (time.time() - total_start) * 1000
        # print(f"总处理时间: {total_time:.2f}ms")

        if True:
            # 绘制图像中心点
            cv2.circle(self.line_image, (int(self.cx), int(self.cy)), 10, (255, 0, 0), 2)
            cv2.circle(self.line_image, (int(center_x), int(self.cy)), 10, (0, 255, 255), 2)
        
        error = center_x - self.cx
        return error

if __name__ == "__main__":
    # 输入视频路径
    input_video_path = "/home/perry_lin/workplace/dog_following_line/sample.mp4"  # 修改为你的输入视频路径
    
    # 输出视频路径
    output_video_path = "/home/perry_lin/workplace/dog_following_line/output.mp4"  # 修改为你的输出视频路径
    binary_output_path = "/home/perry_lin/workplace/dog_following_line/binary_output.mp4"
    
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_video_path}")
        exit()
    
    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {frame_width}x{frame_height}, {fps} FPS, 总帧数: {total_frames}")
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4V编解码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # 修复二值图像输出 - 需要转换为三通道
    binary_out = cv2.VideoWriter(binary_output_path, fourcc, fps, (frame_width, frame_height))
    
    # 初始化LineFollower
    line_follower = LineFollower()
    
    # 处理统计
    processed_frames = 0
    skipped_frames = 0
    start_time = time.time()
    
    # 创建窗口用于显示结果
    cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整帧大小（如果需要）
        if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
            frame = cv2.resize(frame, (frame_width, frame_height))
        
        try:
            # 处理当前帧
            error = line_follower.follow_lines(frame)
            
            # 获取处理后的图像
            processed_frame = line_follower.line_image.copy()
            
            # 修复二值图像输出 - 转换为三通道
            if line_follower.binary_image is not None:
                # 将单通道二值图像转换为三通道
                binary_processed_frame = cv2.cvtColor(line_follower.binary_image, cv2.COLOR_GRAY2BGR)
            else:
                # 如果没有二值图像，创建一个黑色帧
                binary_processed_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # 添加处理信息
            cv2.putText(processed_frame, f"Frame: {processed_frames+1}/{total_frames}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Error: {error:.2f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 在二值图像上添加信息
            cv2.putText(binary_processed_frame, f"Frame: {processed_frames+1}/{total_frames}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(binary_processed_frame, f"Error: {error:.2f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 写入输出视频
            out.write(processed_frame)
            binary_out.write(binary_processed_frame)
            
            # 显示处理进度
            processed_frames += 1
            if processed_frames % 10 == 0:
                elapsed = time.time() - start_time
                fps = processed_frames / elapsed
                remaining = (total_frames - processed_frames) / fps if fps > 0 else 0
                print(f"已处理: {processed_frames}/{total_frames} 帧 | "
                      f"速度: {fps:.1f} FPS | "
                      f"剩余时间: {remaining:.1f}秒")
            
            # 实时显示处理结果
            cv2.imshow("Processed Frame", processed_frame)
            cv2.imshow("Binary Image", binary_processed_frame)
            
            # 按q键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"处理帧 {processed_frames+1} 时出错: {str(e)}")
            skipped_frames += 1
            
            # 写入原始帧（或错误信息帧）
            error_frame = frame.copy()
            cv2.putText(error_frame, f"ERROR: {str(e)}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(error_frame)
            
            # 写入二值图像错误帧（黑色背景）
            binary_error_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(binary_error_frame, f"ERROR: {str(e)}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            binary_out.write(binary_error_frame)
            
            # 显示错误帧
            cv2.imshow("Processed Frame", error_frame)
            cv2.imshow("Binary Image", binary_error_frame)
            
            # 按q键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            processed_frames += 1
            continue
    
    # 释放资源
    cap.release()
    out.release()
    binary_out.release()
    cv2.destroyAllWindows()
    
    # 打印处理摘要
    elapsed_time = time.time() - start_time
    avg_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
    
    print("\n处理完成!")
    print(f"总帧数: {total_frames}")
    print(f"成功处理: {processed_frames - skipped_frames} 帧")
    print(f"跳过: {skipped_frames} 帧")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均处理速度: {avg_fps:.2f} FPS")
    print(f"输出视频已保存至: {output_video_path}")
    print(f"二值图像视频已保存至: {binary_output_path}")