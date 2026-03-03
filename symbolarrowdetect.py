import cv2
import numpy as np
from collections import deque
from picamera2 import Picamera2
import time

class ShapeRecognizer:
    """Recognizes shapes using Raspberry Pi Camera"""
    
    def __init__(self):
        self.points = deque(maxlen=500)
        self.shape_threshold = 0.05
        
    def detect_shapes(self, frame):
        """Detect shapes in the frame using contour detection"""
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for different color ranges to detect all colored shapes
        # Red (0-10 and 170-180)
        lower_red1 = np.array([0, 100, 100], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 100, 100], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # Yellow (15-35)
        lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        
        # Blue (100-130)
        lower_blue = np.array([100, 100, 100], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)
        
        # Purple (125-155)
        lower_purple = np.array([125, 50, 50], dtype=np.uint8)
        upper_purple = np.array([155, 255, 255], dtype=np.uint8)
        
        # Teal/Cyan (80-100)
        lower_teal = np.array([80, 100, 100], dtype=np.uint8)
        upper_teal = np.array([100, 255, 255], dtype=np.uint8)
        
        # Orange (5-25)
        lower_orange = np.array([5, 100, 100], dtype=np.uint8)
        upper_orange = np.array([25, 255, 255], dtype=np.uint8)
        
        # Combine all masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask4 = cv2.inRange(hsv, lower_blue, upper_blue)
        mask5 = cv2.inRange(hsv, lower_purple, upper_purple)
        mask6 = cv2.inRange(hsv, lower_teal, upper_teal)
        mask7 = cv2.inRange(hsv, lower_orange, upper_orange)
        
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
        mask = cv2.bitwise_or(mask, mask4)
        mask = cv2.bitwise_or(mask, mask5)
        mask = cv2.bitwise_or(mask, mask6)
        mask = cv2.bitwise_or(mask, mask7)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_shapes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip very small contours
            if area < 500:
                continue
            
            # Approximate the contour to reduce noise
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            shape_name = self.identify_shape(approx, contour)
            
            if shape_name:
                detected_shapes.append({
                    'name': shape_name,
                    'contour': contour,
                    'approx': approx,
                    'area': area
                })
        
        return detected_shapes, mask
    
    def identify_shape(self, approx, contour):
        """Identify the shape based on number of vertices and properties"""
        
        vertices = len(approx)
        
        # PLUS/CROSS - Check FIRST before other shapes (has many vertices)
        if self.is_plus_cross(contour):
            return "PLUS ➕"
        
        # ARROW - Check for arrow-like shape (6-12 vertices)
        if self.is_arrow(approx):
            direction = self.get_arrow_direction(contour, approx)
            if direction:
                return f"ARROW {direction}"
            else:
                return "ARROW →"
        
        # STAR - Check for star shape (10+ vertices with alternating pattern)
        if self.is_star(approx):
            return "STAR ⭐"
        
        # CIRCLE - Round shape (check before ellipse)
        if self.is_circle(contour):
            # Check if it's a pie/sector (partial circle)
            if self.is_pie_or_sector(contour):
                return "PIE/SECTOR 🥧"
            else:
                return "CIRCLE ⭕"
        
        # ELLIPSE/OVAL - Elongated circle (only if very circular)
        if self.is_ellipse(contour):
            return "OVAL 🔴"
        
        # HEXAGON - 6 vertices
        if vertices == 6:
            return "HEXAGON ⬡"
        
        # TRIANGLE - 3 vertices
        if vertices == 3:
            return "TRIANGLE △"
        
        # SQUARE/DIAMOND - 4 vertices
        if vertices == 4:
            # Check if it's a trapezium (unequal sides)
            if self.is_trapezium(approx):
                return "TRAPEZIUM ⊢"
            # Check if it's a diamond (rotated square)
            elif self.is_diamond(approx):
                return "DIAMOND 🔶"
            else:
                return "RECTANGLE ▭"
        
        # PENTAGON - 5 vertices
        if vertices == 5:
            return "PENTAGON ⬠"
        
        # Star or other polygon
        if vertices > 6:
            return f"POLYGON ({vertices} sides)"
        
        return None
    
    def is_circle(self, contour):
        """Check if contour is a circle"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return False
        
        # Circularity = 4π * Area / Perimeter²
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Circle has circularity close to 1
        return circularity > 0.7
    
    def is_trapezium(self, approx):
        """Check if quadrilateral is a trapezium (parallel sides)"""
        
        if len(approx) != 4:
            return False
        
        # Get the 4 points
        pts = approx.reshape(4, 2)
        
        # Sort points by x coordinate
        pts = pts[np.argsort(pts[:, 0])]
        
        # Get left and right pairs
        left_pts = pts[:2]
        right_pts = pts[2:]
        
        # Sort each pair by y coordinate
        left_pts = left_pts[np.argsort(left_pts[:, 1])]
        right_pts = right_pts[np.argsort(right_pts[:, 1])]
        
        # Calculate slopes
        if left_pts[1][0] - left_pts[0][0] == 0:
            left_slope = float('inf')
        else:
            left_slope = (left_pts[1][1] - left_pts[0][1]) / (left_pts[1][0] - left_pts[0][0])
        
        if right_pts[1][0] - right_pts[0][0] == 0:
            right_slope = float('inf')
        else:
            right_slope = (right_pts[1][1] - right_pts[0][1]) / (right_pts[1][0] - right_pts[0][0])
        
        # If slopes are different, it's a trapezium (only one pair of parallel sides)
        return abs(left_slope - right_slope) > 0.5
    
    def is_star(self, approx):
        """Check if shape is a star (10+ vertices with alternating distances)"""
        vertices = len(approx)
        
        # Stars typically have 10 or more vertices
        if vertices < 10:
            return False
        
        # Get centroid
        M = cv2.moments(approx.astype('float32'))
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            return False
        
        # Calculate distances from centroid to each vertex
        distances = []
        for point in approx:
            x, y = point[0]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            distances.append(dist)
        
        # Check for alternating pattern (inner and outer points)
        if len(distances) < 10:
            return False
        
        # If distances alternate between small and large, it's a star
        avg_dist = np.mean(distances)
        alternating = 0
        
        for i in range(len(distances) - 1):
            if (distances[i] < avg_dist and distances[i+1] > avg_dist) or \
               (distances[i] > avg_dist and distances[i+1] < avg_dist):
                alternating += 1
        
        return alternating > len(distances) * 0.6
    
    def is_diamond(self, approx):
        """Check if 4-vertex shape is a diamond (rotated square)"""
        if len(approx) != 4:
            return False
        
        pts = approx.reshape(4, 2)
        
        # Calculate distances between consecutive points
        distances = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            distances.append(dist)
        
        # Diamond has roughly equal side lengths (like a square)
        avg_dist = np.mean(distances)
        is_equal_sides = all(abs(d - avg_dist) < avg_dist * 0.2 for d in distances)
        
        if not is_equal_sides:
            return False
        
        # Check if it's rotated (diagonal orientation)
        # Diamond should have a point at top/bottom
        pts_sorted = pts[np.argsort(pts[:, 1])]  # Sort by y
        
        # Top and bottom points should be close to x-center
        top_x = pts_sorted[0][0]
        bottom_x = pts_sorted[-1][0]
        center_x = np.mean(pts[:, 0])
        
        return abs(top_x - center_x) < 20 and abs(bottom_x - center_x) < 20
    
    def is_plus_cross(self, contour):
        """Check if shape is a plus or cross"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Get the area of the shape and bounding box
        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        
        if bbox_area == 0:
            return False
        
        # Plus/cross shapes have lower solidity (not completely filling bounding box)
        solidity = contour_area / bbox_area
        
        # Plus/cross shapes typically have:
        # 1. Roughly square bounding box (aspect ratio ~1)
        # 2. Low solidity (30-70% of bounding box)
        # 3. Many vertices (8+)
        
        aspect_ratio = float(w) / h if h > 0 else 0
        has_square_bbox = 0.7 < aspect_ratio < 1.3
        has_low_solidity = 0.3 < solidity < 0.75
        
        return has_square_bbox and has_low_solidity
    
    def is_pie_or_sector(self, contour):
        """Check if shape is a pie chart or sector (partial circle)"""
        # Pie/sector has curved edge and straight edges meeting at center
        
        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        if hull_area == 0:
            return False
        
        solidity = contour_area / hull_area
        
        # Pie/sector typically has lower solidity (not completely filled)
        return 0.6 < solidity < 0.95
    
    def is_ellipse(self, contour):
        """Check if shape is an oval/ellipse (elongated circle)"""
        if len(contour) < 5:
            return False
        
        try:
            # Fit ellipse to contour
            ellipse = cv2.fitEllipse(contour)
            
            # Get ellipse properties
            (center, (width, height), angle) = ellipse
            
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            
            if hull_area == 0:
                return False
            
            solidity = contour_area / hull_area
            
            # Ellipse should have very high solidity (>0.9, mostly filled)
            # and be quite circular (aspect ratio 0.5-2.0)
            # BUT must NOT be a plus/cross
            if solidity > 0.9:
                if height > 0:
                    aspect_ratio = width / height
                    # Only true ellipses/ovals - not plus/cross shapes
                    return 0.5 < aspect_ratio < 2.0 and not self.is_plus_cross(contour)
            
            return False
        except:
            return False
    
    def is_arrow(self, approx):
        """Check if shape looks like an arrow"""
        
        # Arrows typically have more vertices and a pointed end
        vertices = len(approx)
        
        if vertices < 6 or vertices > 12:
            return False
        
        # Check if there's a sharp point (high curvature)
        # This is a simplified check
        return True
    
    def get_arrow_direction(self, contour, approx):
        """Determine the direction of an arrow (up, down, left, right)"""
        
        # Get the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            return None
        
        # Find the point furthest from the centroid (likely the arrow tip)
        max_dist = 0
        tip_point = None
        
        for point in approx:
            x, y = point[0]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist > max_dist:
                max_dist = dist
                tip_point = (x, y)
        
        if tip_point is None:
            return None
        
        # Determine direction based on tip position relative to centroid
        dx = tip_point[0] - cx
        dy = tip_point[1] - cy
        
        # Calculate angle
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Determine direction (with some tolerance)
        if -45 <= angle <= 45:
            return "RIGHT →"
        elif 45 < angle <= 135:
            return "DOWN ↓"
        elif -135 <= angle < -45:
            return "UP ↑"
        else:
            return "LEFT ←"
    
    def process_frame(self, frame):
        """Process frame and detect shapes"""
        
        detected_shapes, mask = self.detect_shapes(frame)
        
        # Draw detected shapes
        for shape in detected_shapes:
            contour = shape['contour']
            approx = shape['approx']
            name = shape['name']
            
            # Draw contour
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            
            # Draw vertices
            for point in approx:
                x, y = point[0]
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # Get bounding rectangle for label
            x, y, w, h = cv2.boundingRect(contour)
            
            # Put shape name
            cv2.putText(frame, name, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Area: {shape['area']:.0f}", (x, y + h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw arrow direction indicator if it's an arrow
            if "ARROW" in name:
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw circle at centroid
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
                    
                    # Draw line from centroid to tip
                    direction = self.get_arrow_direction(contour, approx)
                    if direction:
                        # Find tip point
                        max_dist = 0
                        tip_point = None
                        for point in approx:
                            px, py = point[0]
                            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                            if dist > max_dist:
                                max_dist = dist
                                tip_point = (int(px), int(py))
                        
                        if tip_point:
                            cv2.arrowedLine(frame, (cx, cy), tip_point, (255, 0, 255), 2, tipLength=0.3)
        
        return frame, detected_shapes, mask


def main():
    """Main function to run shape recognition from Raspberry Pi Camera"""
    print("🎥 Starting Shape Recognition (Raspberry Pi Camera)...")
    print("Recognizes: Circle, Triangle, Rectangle, Trapezium, Pentagon, Arrow, Diamond, Plus, Star, Oval")
    print("Press 'q' to quit")
    print("Press 's' to show/hide detection mask\n")
    
    try:
        # Initialize Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        
        print("✅ Raspberry Pi Camera initialized successfully")
        print("Resolution: 640x480")
        time.sleep(2)  # Give camera time to warm up
        
    except Exception as e:
        print(f"❌ Error: Failed to initialize camera: {e}")
        print("Make sure:")
        print("  1. Camera is connected to CSI port")
        print("  2. Camera is enabled: sudo raspi-config (Interfacing -> Camera)")
        print("  3. picamera2 is installed: sudo apt install -y python3-picamera2")
        return
    
    recognizer = ShapeRecognizer()
    show_mask = False
    
    try:
        while True:
            # Capture frame from camera
            frame = picam2.capture_array()
            
            if frame is None:
                print("❌ Error: Failed to read frame")
                break
            
            # Convert XRGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Flip frame for natural view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            annotated_frame, shapes, mask = recognizer.process_frame(frame)
            
            # Display info
            cv2.putText(annotated_frame, "Shape Recognition (RPi)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Shapes detected: {len(shapes)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if shapes:
                for i, shape in enumerate(shapes):
                    y_offset = 110 + (i * 35)
                    cv2.putText(annotated_frame, f"{i+1}. {shape['name']}", 
                               (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(annotated_frame, "Show a shape...", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(annotated_frame, "Press 'q' to quit | 's' for mask", 
                       (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Shape Recognition', annotated_frame)
            
            if show_mask:
                cv2.imshow('Detection Mask', mask)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Shape recognition stopped")
                break
            elif key == ord('s'):
                show_mask = not show_mask
                print(f"Mask display: {'ON' if show_mask else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        # Cleanup
        picam2.stop()
        cv2.destroyAllWindows()
        print("✅ Camera cleaned up")


if __name__ == "__main__":
    main()
