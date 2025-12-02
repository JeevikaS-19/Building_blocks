import cv2
import mediapipe as mp
import time
import pymunk
import pymunk.pygame_util
import math
import numpy as np


class Shape:
    def __init__(self, points, color=(255, 255, 255)):
        self.points = points  # List of (x, y) tuples
        self.color = color
        self.body = None
        self.poly = None
        self.z_rotation = 0
        self.x_rotation = 0
        self.y_rotation = 0
        self.velocity = (0, 0)


class HandTracker:
    """Hand tracking class using MediaPipe"""
    
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the HandTracker
        
        Args:
            mode: Static image mode (False for video)
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=True):
        """
        Find hands in the image
        
        Args:
            img: Input image (BGR format)
            draw: Whether to draw hand landmarks
            
        Returns:
            Image with drawn landmarks (if draw=True)
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    pass
        
        return img
    
    def find_position(self, img, hand_no=0):
        """
        Find positions of hand landmarks
        
        Args:
            img: Input image
            hand_no: Hand index (0 for first hand, 1 for second)
            
        Returns:
            List of landmark positions [(id, x, y, z), ...]
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # lm.z is relative to image width, not pixels. 
                    # But for tilt calculation, relative is fine.
                    landmark_list.append((id, cx, cy, lm.z))
        
        return landmark_list
    
    def get_hand_label(self, hand_no=0):
        """
        Get the label of the hand (Left or Right)
        
        Args:
            hand_no: Hand index
            
        Returns:
            'Left' or 'Right' or None
        """
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_handedness):
                hand_label = self.results.multi_handedness[hand_no].classification[0].label
                return hand_label
        return None
    
    def is_pinching(self, landmark_list, threshold=40):
        """
        Detect if thumb tip and index finger tip are close (pinching gesture)
        
        Args:
            landmark_list: List of landmarks for a hand
            threshold: Distance threshold in pixels to consider as pinching
            
        Returns:
            Boolean indicating if pinching is detected
        """
        if len(landmark_list) >= 9:
            # Landmark 4 = thumb tip, Landmark 8 = index finger tip
            thumb_tip = landmark_list[4]
            index_tip = landmark_list[8]
            
            import math
            distance = math.sqrt(
                (thumb_tip[1] - index_tip[1])**2 + 
                (thumb_tip[2] - index_tip[2])**2
            )
            
            return distance < threshold
        return False
    
    def is_thumbs_up(self, landmark_list):
        """
        Detect thumbs up gesture
        
        Args:
            landmark_list: List of landmarks for a hand
            
        Returns:
            Boolean indicating if thumbs up is detected
        """
        if len(landmark_list) >= 21:
            # Landmark 4 = thumb tip, Landmark 3 = thumb IP joint, Landmark 2 = thumb MCP
            # Landmark 8 = index tip, Landmark 6 = index PIP
            # Landmark 12 = middle tip, Landmark 10 = middle PIP
            # Landmark 16 = ring tip, Landmark 14 = ring PIP
            # Landmark 20 = pinky tip, Landmark 18 = pinky PIP
            
            thumb_tip = landmark_list[4]
            thumb_ip = landmark_list[3]
            thumb_mcp = landmark_list[2]
            
            index_tip = landmark_list[8]
            index_pip = landmark_list[6]
            
            middle_tip = landmark_list[12]
            middle_pip = landmark_list[10]
            
            ring_tip = landmark_list[16]
            ring_pip = landmark_list[14]
            
            pinky_tip = landmark_list[20]
            pinky_pip = landmark_list[18]
            
            # 1. Thumb should be pointing UP
            # Tip should be significantly higher (lower y value) than IP joint and MCP
            thumb_is_up = (thumb_tip[2] < thumb_ip[2] - 10) and (thumb_ip[2] < thumb_mcp[2] - 10)
            
            # 2. Other fingers should be curled
            # Tips should be lower (higher y value) than their PIP joints
            index_curled = index_tip[2] > index_pip[2]
            middle_curled = middle_tip[2] > middle_pip[2]
            ring_curled = ring_tip[2] > ring_pip[2]
            pinky_curled = pinky_tip[2] > pinky_pip[2]
            
            # Debug print occasionally
            # import random
            # if random.random() < 0.05:
            #     print(f"ThumbUp: {thumb_is_up}, I:{index_curled}, M:{middle_curled}, R:{ring_curled}, P:{pinky_curled}")
            
            return thumb_is_up and index_curled and middle_curled and ring_curled and pinky_curled
        return False
    


def straighten_line(point1, point2, angle_threshold=15):
    """
    Straighten a line to horizontal or vertical if close to those angles
    
    Args:
        point1: (x, y) tuple for first point
        point2: (x, y) tuple for second point
        angle_threshold: Degrees within which to snap to horizontal/vertical
        
    Returns:
        Adjusted point2 to make line horizontal or vertical
    """
    import math
    
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    # Calculate angle in degrees
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    
    # Snap to horizontal if close to 0 or 180 degrees
    if angle < angle_threshold or angle > (180 - angle_threshold):
        return (point2[0], point1[1])  # Same y as point1
    
    # Snap to vertical if close to 90 degrees
    if abs(angle - 90) < angle_threshold:
        return (point1[0], point2[1])  # Same x as point1
    
    # Otherwise return original point
    return point2


def snap_to_existing_points(new_point, existing_lines, threshold=30):
    """
    Snap a new point to existing points if close enough
    
    Args:
        new_point: (x, y) tuple of new point
        existing_lines: List of finalized lines [(pt1, pt2, dist), ...]
        threshold: Distance threshold for snapping
        
    Returns:
        Snapped point or original point
    """
    import math
    
    closest_point = None
    min_dist = float('inf')
    
    # Check all endpoints of existing lines
    for pt1, pt2, _ in existing_lines:
        # Check start point
        dist1 = math.sqrt((new_point[0] - pt1[0])**2 + (new_point[1] - pt1[1])**2)
        if dist1 < threshold and dist1 < min_dist:
            min_dist = dist1
            closest_point = pt1
            
        # Check end point
        dist2 = math.sqrt((new_point[0] - pt2[0])**2 + (new_point[1] - pt2[1])**2)
        if dist2 < threshold and dist2 < min_dist:
            min_dist = dist2
            closest_point = pt2
            
    if closest_point is not None:
        return closest_point
    
    return new_point


def rotate_point(point, center, angle_degrees):
    """
    Rotate a point around a center
    
    Args:
        point: (x, y) tuple
        center: (cx, cy) tuple
        angle_degrees: Angle in degrees
        
    Returns:
        Rotated (x, y) tuple
    """
    import math
    
    angle_rad = math.radians(angle_degrees)
    
    # Translate point to origin
    px, py = point
    cx, cy = center
    
    tx = px - cx
    ty = py - cy
    
    # Rotate
    rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
    ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
    
    # Translate back
    return (int(rx + cx), int(ry + cy))


def rotate_point_3d(point, center, angle_x, angle_y, angle_z):
    """
    Rotate a point in 3D around a center
    """
    import math
    
    # Translate to origin
    x = point[0] - center[0]
    y = point[1] - center[1]
    z = 0 
    
    # Convert to radians
    rad_x = math.radians(angle_x)
    rad_y = math.radians(angle_y)
    rad_z = math.radians(angle_z)
    
    # 1. Rotate around X (Tilt Up/Down)
    y1 = y * math.cos(rad_x) - z * math.sin(rad_x)
    z1 = y * math.sin(rad_x) + z * math.cos(rad_x)
    x1 = x
    
    # 2. Rotate around Y (Tilt Left/Right)
    z2 = z1 * math.cos(rad_y) - x1 * math.sin(rad_y)
    x2 = z1 * math.sin(rad_y) + x1 * math.cos(rad_y)
    y2 = y1
    
    # 3. Rotate around Z (Spin)
    x3 = x2 * math.cos(rad_z) - y2 * math.sin(rad_z)
    y3 = x2 * math.sin(rad_z) + y2 * math.cos(rad_z)
    
    # Translate back
    return (int(x3 + center[0]), int(y3 + center[1]))


def get_unique_points(lines, tolerance=20):
    """
    Map raw points to unique IDs based on proximity.
    Returns:
        points: List of unique (x, y) tuples
        point_map: Dict mapping original (x, y) to unique index
    """
    unique_points = []
    point_map = {}
    
    # Helper to find close point
    def find_close_point(pt):
        for i, unique_pt in enumerate(unique_points):
            dist = math.sqrt((pt[0] - unique_pt[0])**2 + (pt[1] - unique_pt[1])**2)
            if dist < tolerance:
                return i
        return -1

    for pt1, pt2, _ in lines:
        # Check pt1
        idx1 = find_close_point(pt1)
        if idx1 == -1:
            unique_points.append(pt1)
            idx1 = len(unique_points) - 1
        point_map[pt1] = idx1
        
        # Check pt2
        idx2 = find_close_point(pt2)
        if idx2 == -1:
            unique_points.append(pt2)
            idx2 = len(unique_points) - 1
        point_map[pt2] = idx2
            
    return unique_points, point_map

def build_adjacency_graph(lines, tolerance=20):
    """
    Build a graph where nodes are points and edges are lines.
    Returns:
        adj: Dict mapping point_index -> list of (neighbor_index, line_index)
        unique_points: List of unique point coordinates
    """
    unique_points, point_map = get_unique_points(lines, tolerance)
    adj = {i: [] for i in range(len(unique_points))}
    
    for line_idx, (pt1, pt2, _) in enumerate(lines):
        idx1 = point_map[pt1]
        idx2 = point_map[pt2]
        
        if idx1 != idx2: # Ignore zero-length lines
            adj[idx1].append((idx2, line_idx))
            adj[idx2].append((idx1, line_idx))
            
    return adj, unique_points

def find_closed_loop(lines):
    """
    Find a set of lines that form a closed loop where every point has degree 2.
    Returns:
        loop_lines: List of lines forming the loop (or None)
        loop_indices: Indices of these lines in the original list (or None)
        shape_type: String describing the shape (Triangle, Quad, Polygon)
    """
    if len(lines) < 3:
        return None, None, None
        
    adj, unique_points = build_adjacency_graph(lines)
    
    # Find connected components
    visited = set()
    components = []
    
    for i in range(len(unique_points)):
        if i not in visited:
            component_nodes = []
            stack = [i]
            visited.add(i)
            while stack:
                node = stack.pop()
                component_nodes.append(node)
                for neighbor, _ in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append(component_nodes)
            
    # Check each component
    for comp in components:
        if len(comp) < 3:
            continue
            
        # Check degrees
        is_closed_loop = True
        comp_lines_indices = set()
        
        for node in comp:
            # In a simple closed polygon, every node must have degree 2
            # Note: This handles simple polygons. Complex ones with intersections might fail this simple check.
            degree = len(adj[node])
            if degree != 2:
                is_closed_loop = False
                break
            
            # Collect line indices
            for _, line_idx in adj[node]:
                comp_lines_indices.add(line_idx)
        
        if is_closed_loop:
            # We found a loop!
            loop_indices = sorted(list(comp_lines_indices))
            loop_lines = [lines[i] for i in loop_indices]
            
            num_sides = len(loop_lines)
            shape_type = "POLYGON"
            if num_sides == 3:
                shape_type = "TRIANGLE"
            elif num_sides == 4:
                shape_type = "QUAD" # Could be rect, square, or generic quad
                
            return loop_lines, loop_indices, shape_type
            
    return None, None, None

def order_loop_points(lines, tolerance=20):
    """
    Given a set of lines forming a closed loop, return the ordered vertices.
    """
    if not lines:
        return []
        
    # Get unique points to handle connectivity
    unique_points, point_map = get_unique_points(lines, tolerance)
    
    # Build adjacency for this specific set of lines
    # map point_idx -> list of (neighbor_idx, line_idx)
    adj = {i: [] for i in range(len(unique_points))}
    for line_idx, (pt1, pt2, _) in enumerate(lines):
        idx1 = point_map[pt1]
        idx2 = point_map[pt2]
        adj[idx1].append(idx2)
        adj[idx2].append(idx1)
        
    # Traverse
    ordered_indices = []
    start_node = 0 # Start with first unique point
    curr = start_node
    visited = {curr}
    ordered_indices.append(curr)
    
    # We expect a simple cycle
    while len(ordered_indices) < len(unique_points):
        # Find unvisited neighbor
        found = False
        for neighbor in adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                ordered_indices.append(neighbor)
                curr = neighbor
                found = True
                break
        if not found:
            break
            
    # Convert indices back to points
    ordered_points = [unique_points[i] for i in ordered_indices]
    return ordered_points

def adjust_rectangle_if_applicable(lines, pixels_per_cm):
    """
    If the lines form a 4-sided shape that looks like a rectangle, adjust it.
    Otherwise return the lines as-is (but ordered).
    
    Args:
        lines: List of 4 lines
        pixels_per_cm: Calibration value
        
    Returns:
        adjusted_lines: List of 4 lines (adjusted or original)
        shape_type: "SQUARE", "RECTANGLE", or "QUAD"
    """
    if len(lines) != 4:
        return lines, "POLYGON"
        
    # Per user request: "when it's a polygon, keep the shape as it was drawn, don't auto-adjust"
    # We will disable the auto-adjustment for now and just return the original lines as a QUAD.
    return lines, "QUAD"
    
    # Original logic commented out below:
    """
    # Extract points
    points = []
    for pt1, pt2, _ in lines:
        points.append(pt1)
        points.append(pt2)
        
    # Get bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width_px = max_x - min_x
    height_px = max_y - min_y
    
    # Check if it's axis-aligned enough to be a rectangle
    # We can check if the area of bounding box is close to area of polygon
    # Or simpler: check if width/height ratio is reasonable and lines are roughly horizontal/vertical
    # For now, let's use the existing logic but be more permissive since we already know it's a closed loop
    
    if pixels_per_cm:
        width_cm = width_px / pixels_per_cm
        height_cm = height_px / pixels_per_cm
        
        diff_cm = abs(width_cm - height_cm)
        
        # Heuristic: If it's a quad, let's see if we should snap it to a rect/square
        # We'll assume if the user drew a closed 4-sided loop, they might want a rect
        # UNLESS it's clearly a diamond or trapezium. 
        # For this iteration, let's ALWAYS try to snap to rect/square for 4 sides
        # as that preserves the original behavior for boxes.
        
        shape_type = ""
        new_width_cm = 0
        new_height_cm = 0
        
        if diff_cm < 2.0:
            shape_type = "SQUARE"
            avg_side = (width_cm + height_cm) / 2
            new_width_cm = avg_side
            new_height_cm = avg_side
        else:
            shape_type = "RECTANGLE"
            new_width_cm = width_cm
            new_height_cm = height_cm
            
        # Reconstruct perfect rect
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        half_w = int(new_width_cm * pixels_per_cm / 2)
        half_h = int(new_height_cm * pixels_per_cm / 2)
        
        tl = (center_x - half_w, center_y - half_h)
        tr = (center_x + half_w, center_y - half_h)
        br = (center_x + half_w, center_y + half_h)
        bl = (center_x - half_w, center_y + half_h)
        
        new_lines = [
            (tl, tr, new_width_cm),
            (tr, br, new_height_cm),
            (br, bl, new_width_cm),
            (bl, tl, new_height_cm)
        ]
        return new_lines, shape_type
        
    return lines, "QUAD"
    """




def main():
    """Main function to run the hand gesture drawing application"""
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera 0 failed, trying Camera 1...")
        cap = cv2.VideoCapture(1)
        
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height
    
    # Initialize hand tracker
    tracker = HandTracker(max_hands=2)
    
    # For FPS calculation
    prev_time = 0
    
    # Calibration: Average adult hand width is ~8.5 cm
    # We'll use hand width to estimate pixels per cm
    pixels_per_cm = None
    AVERAGE_HAND_WIDTH_CM = 8.5
    
    # Locked points for current line
    locked_point_1 = None  # Left index locked position
    locked_point_2 = None  # Right index locked position
    left_was_pinching = False
    right_was_pinching = False
    
    # Finalized lines storage
    finalized_lines = []  # List of (point1, point2, distance_cm) tuples
    shapes = [] # List of Shape objects
    thumbs_up_was_detected = False
    
    # Rotation Mode
    rotation_mode = False
    rotation_center = None
    active_shape = None # The shape currently being rotated
    base_rotation_angle = 0
    current_rotation_angle = 0
    initial_thumb_index_angle = None
    shape_offsets = [] # Store offsets from center
    
    # Physics System
    physics_enabled = True  # Physics ON by default
    space = pymunk.Space()
    space.gravity = (0, 900)  # Downward gravity
    
    # Create static boundary walls
    screen_width, screen_height = 1280, 720
    static_body = space.static_body
    
    # Bottom wall
    bottom_wall = pymunk.Segment(static_body, (0, screen_height), (screen_width, screen_height), 5)
    bottom_wall.elasticity = 0.7
    bottom_wall.friction = 0.5
    space.add(bottom_wall)
    
    # Left wall
    left_wall = pymunk.Segment(static_body, (0, 0), (0, screen_height), 5)
    left_wall.elasticity = 0.7
    left_wall.friction = 0.5
    space.add(left_wall)
    
    # Right wall
    right_wall = pymunk.Segment(static_body, (screen_width, 0), (screen_width, screen_height), 5)
    right_wall.elasticity = 0.7
    right_wall.friction = 0.5
    space.add(right_wall)
    
    # Top wall
    top_wall = pymunk.Segment(static_body, (0, 0), (screen_width, 0), 5)
    top_wall.elasticity = 0.7
    top_wall.friction = 0.5
    space.add(top_wall)
    
    # Track previous thumb position for velocity calculation
    prev_thumb_pos = None
    prev_thumb_time = None
    
    print("Hand Gesture Drawing Application")
    print("=" * 50)
    print("Instructions:")
    print("- Show both hands to the camera")
    print("- Pinch left thumb + index to SET Point 1 (Start)")
    print("- Pinch right thumb + index to SET Point 2 (End)")
    print("- Lines auto-straighten to horizontal/vertical")
    print("- Points snap to existing lines (magnetic)")
    print("- Thumbs up gesture to FINALIZE line and start new one")
    print("- Auto-detects Square/Rectangle (4 lines)")
    print("- Rotation Mode activates after shape detection")
    print("- Press 'u' to undo last line")
    print("- Press 'c' to clear all lines")
    print("- Press 'q' to quit")
    print("=" * 50)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture video")
            break
        
        # Flip image for mirror effect
        img = cv2.flip(img, 1)
        
        # Find hands
        img = tracker.find_hands(img)
        
        # Store index finger positions and pinch states
        left_index = None
        right_index = None
        left_is_pinching = False
        right_is_pinching = False
        thumbs_up_detected = False
        
        # Process each detected hand
        if tracker.results.multi_hand_landmarks:
            for hand_no in range(len(tracker.results.multi_hand_landmarks)):
                # Get hand label
                hand_label = tracker.get_hand_label(hand_no)
                
                # Get landmark positions
                landmark_list = tracker.find_position(img, hand_no)
                
                if len(landmark_list) > 0:
                    # Calibrate using hand width (wrist to pinky base)
                    # Landmark 0 = wrist, Landmark 17 = pinky base
                    if len(landmark_list) >= 18:
                        wrist = landmark_list[0]
                        pinky_base = landmark_list[17]
                        import math
                        hand_width_pixels = math.sqrt(
                            (pinky_base[1] - wrist[1])**2 + 
                            (pinky_base[2] - wrist[2])**2
                        )
                        # Update calibration (running average)
                        current_ppc = hand_width_pixels / AVERAGE_HAND_WIDTH_CM
                        if pixels_per_cm is None:
                            pixels_per_cm = current_ppc
                        else:
                            # Smooth the calibration
                            pixels_per_cm = 0.9 * pixels_per_cm + 0.1 * current_ppc
                    
                    # Index finger tip is landmark 8
                    index_finger = landmark_list[8]
                    x, y = index_finger[1], index_finger[2]
                    
                    # Check if pinching
                    is_pinching = tracker.is_pinching(landmark_list)
                    
                    # Check for thumbs up gesture
                    is_thumbs_up = tracker.is_thumbs_up(landmark_list)
                    if is_thumbs_up:
                        thumbs_up_detected = True
                    
                    # Store position and pinch state based on hand label
                    if hand_label == "Left":
                        left_index = (x, y)
                        left_is_pinching = is_pinching
                        
                        # ROTATION MODE LOGIC
                        if rotation_mode:
                            # Use Thumb (4) as center and Index (8) for angle
                            t_x, t_y = landmark_list[4][1], landmark_list[4][2]
                            i_x, i_y = landmark_list[8][1], landmark_list[8][2]
                            
                            # Update Rotation Center to Thumb Position
                            rotation_center = (t_x, t_y)
                            
                            # Draw rotation control (Thumb and Index)
                            cv2.line(img, (t_x, t_y), (i_x, i_y), (0, 0, 255), 2)
                            cv2.circle(img, (t_x, t_y), 10, (0, 0, 255), cv2.FILLED) # Pivot (Thumb)
                            cv2.circle(img, (i_x, i_y), 10, (0, 255, 255), cv2.FILLED) # Handle (Index)
                            
                            # Calculate angle of the "handle" (vector from thumb to index)
                            import math
                            # Angle relative to horizontal axis
                            angle = math.degrees(math.atan2(i_y - t_y, i_x - t_x))
                            
                            if initial_thumb_index_angle is None:
                                initial_thumb_index_angle = angle
                                print(f"Rotation started. Initial angle: {angle:.1f}")
                            
                            # Delta angle (how much the user rotated their hand)
                            delta_angle = angle - initial_thumb_index_angle
                            
                            # Update current rotation
                            current_rotation_angle = base_rotation_angle + delta_angle
                            
                            # Update active shape rotation (Z-axis only, no 3D tilt)
                            if active_shape:
                                active_shape.z_rotation = current_rotation_angle
                                active_shape.x_rotation = 0  # No X tilt
                                active_shape.y_rotation = 0  # No Y tilt
                            
                            cv2.putText(img, f"Rotate: {int(current_rotation_angle)} deg", 
                                       (t_x - 50, t_y - 40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                        # DRAWING MODE LOGIC
                        elif not rotation_mode:
                            # Draw Thumb and Index explicitly since we disabled full hand drawing
                            t_x, t_y = landmark_list[4][1], landmark_list[4][2]
                            i_x, i_y = landmark_list[8][1], landmark_list[8][2]
                            cv2.circle(img, (t_x, t_y), 8, (200, 200, 200), cv2.FILLED) # Thumb
                            
                            # Lock point on pinch (when transitioning from not pinching to pinching)
                            if is_pinching and not left_was_pinching:
                                # Snap to existing points
                                snapped_point = snap_to_existing_points((x, y), finalized_lines)
                                locked_point_1 = snapped_point
                                print(f"Point 1 LOCKED at: {locked_point_1}")
                            
                            # Draw current position (Index)
                            color = (0, 200, 0) if is_pinching else (0, 255, 0)
                            cv2.circle(img, (x, y), 15, color, cv2.FILLED)
                            cv2.putText(img, "1", (x - 10, y + 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Draw gesture indicators
                        if is_pinching:
                            cv2.putText(img, "PINCH", (x - 30, y - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if is_thumbs_up:
                            cv2.putText(img, "THUMBS UP!", (x - 50, y - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                    elif hand_label == "Right":
                        right_index = (x, y)
                        right_is_pinching = is_pinching
                        
                        if not rotation_mode:
                            # Draw Thumb and Index explicitly
                            t_x, t_y = landmark_list[4][1], landmark_list[4][2]
                            cv2.circle(img, (t_x, t_y), 8, (200, 200, 200), cv2.FILLED) # Thumb
                            
                            # Lock point on pinch (when transitioning from not pinching to pinching)
                            if is_pinching and not right_was_pinching:
                                # Snap to existing points
                                snapped_point = snap_to_existing_points((x, y), finalized_lines)
                                locked_point_2 = snapped_point
                                print(f"Point 2 LOCKED at: {locked_point_2}")
                            
                            # Draw current position
                            color = (0, 0, 200) if is_pinching else (0, 0, 255)
                            cv2.circle(img, (x, y), 15, color, cv2.FILLED)
                            cv2.putText(img, "2", (x - 10, y + 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        
        # Update previous pinch states
        if left_index:
            left_was_pinching = left_is_pinching
        if right_index:
            right_was_pinching = right_is_pinching 
        
        # Handle thumbs up gesture to finalize current line
        if thumbs_up_detected and not thumbs_up_was_detected:
            if locked_point_1 is not None and locked_point_2 is not None:
                # Straighten the line before finalizing
                straightened_point_2 = straighten_line(locked_point_1, locked_point_2)
                
                # Calculate distance for finalized line
                import math
                distance_pixels = math.sqrt((straightened_point_2[0] - locked_point_1[0])**2 + 
                                           (straightened_point_2[1] - locked_point_1[1])**2)
                distance_cm = None
                if pixels_per_cm is not None and pixels_per_cm > 0:
                    distance_cm = distance_pixels / pixels_per_cm
                
                # Save the finalized line
                finalized_lines.append((locked_point_1, straightened_point_2, distance_cm))
                print(f"Line FINALIZED: {len(finalized_lines)} total lines")
                
                # Check for closed shape detection
                loop_lines, loop_indices, shape_type = find_closed_loop(finalized_lines)
                
                if loop_lines:
                    # If it's a quad, try to adjust it to a rectangle/square
                    final_shape_lines = loop_lines
                    if shape_type == "QUAD":
                        adjusted_lines, specific_type = adjust_rectangle_if_applicable(loop_lines, pixels_per_cm)
                        final_shape_lines = adjusted_lines
                        shape_type = specific_type
                    
                    print(f"Shape Detected: {shape_type}")
                    
                    # Create Shape object
                    # Extract ordered points from lines
                    ordered_points = []
                    # We need to order them correctly. The loop_lines might not be in order.
                    # But find_closed_loop returns lines based on graph traversal, so they should be connected.
                    # Let's trace the path from the first line
                    
                    # Simple approach: just take points from lines. 
                    # Since adjust_rectangle_if_applicable returns ordered lines for quads, we are good there.
                    # For general polygons, we might need to ensure order.
                    # Let's assume for now loop_lines are somewhat ordered or we just take vertices.
                    
                    # Better: Re-extract points from the lines to ensure they form a sequence
                    # For adjusted quads, they are definitely ordered (tl->tr->br->bl)
                    if shape_type in ["SQUARE", "RECTANGLE"]:
                         ordered_points = [final_shape_lines[0][0], final_shape_lines[1][0], final_shape_lines[2][0], final_shape_lines[3][0]]
                    else:
                        # For generic polygons, use the helper to get ordered vertices
                        ordered_points = order_loop_points(final_shape_lines)

                    active_shape = Shape(ordered_points)
                    shapes.append(active_shape)
                    
                    # Remove used lines from finalized_lines
                    # We must remove by index, starting from largest to avoid shifting issues
                    for idx in sorted(loop_indices, reverse=True):
                        finalized_lines.pop(idx)
                    
                    # Show notification
                    cv2.putText(img, f"{shape_type} DETECTED!", 
                               (img.shape[1]//2 - 150, img.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.imshow("Hand Gesture Drawing", img)
                    cv2.waitKey(1000) # Show for 1 second
                    
                    # ENTER ROTATION MODE
                    rotation_mode = True
                    base_rotation_angle = 0
                    current_rotation_angle = 0
                    initial_thumb_index_angle = None
                    
                    # Calculate center of shape (initial pivot)
                    xs = [p[0] for p in active_shape.points]
                    ys = [p[1] for p in active_shape.points]
                    initial_center_x = (min(xs) + max(xs)) // 2
                    initial_center_y = (min(ys) + max(ys)) // 2
                    
                    # Calculate OFFSETS from center for each point
                    # This allows us to reconstruct the shape around ANY new center (like the thumb)
                    shape_offsets = []
                    for pt in active_shape.points:
                        off = (pt[0] - initial_center_x, pt[1] - initial_center_y)
                        shape_offsets.append(off)
                        
                    print(f"Entering Rotation Mode. Shape pinned to Thumb.")
                    
                    # Add a frame counter to prevent immediate exit
                    rotation_entry_frame = 0
                
                # Clear current locked points for next line
                locked_point_1 = None
                locked_point_2 = None
        
        # Increment rotation entry frame counter
        if rotation_mode:
            if 'rotation_entry_frame' in locals():
                rotation_entry_frame += 1
        
        # Handle Thumbs Up to EXIT Rotation Mode (but not immediately after entering)
        can_exit_rotation = rotation_mode and ('rotation_entry_frame' not in locals() or rotation_entry_frame > 30)
        if can_exit_rotation and thumbs_up_detected and not thumbs_up_was_detected:
            rotation_mode = False
            print("Rotation Mode Exited. Shape Finalized.")
            
            # Re-calculate final positions based on last known thumb/index
            if rotation_center is not None and active_shape:
                 new_points = []
                 for off in shape_offsets:
                     # Rotate offset
                     r_off = rotate_point((off[0], off[1]), (0,0), current_rotation_angle)
                     # Add to new center
                     new_pt = (rotation_center[0] + r_off[0], rotation_center[1] + r_off[1])
                     new_points.append(new_pt)
                 
                 active_shape.points = new_points
                 
                 # CREATE PHYSICS BODY
                 if physics_enabled:
                     # Calculate centroid
                     cx = sum(p[0] for p in new_points) / len(new_points)
                     cy = sum(p[1] for p in new_points) / len(new_points)
                     centroid = (cx, cy)
                     
                     # Points relative to centroid
                     relative_points = [(p[0]-cx, p[1]-cy) for p in new_points]
                     
                     # Validate and create physics body
                     import math
                     mass = 1
                     try:
                         moment = pymunk.moment_for_poly(mass, relative_points)
                         # Check if moment is valid
                         if math.isnan(moment) or math.isinf(moment) or moment <= 0:
                             # Use box approximation as fallback
                             moment = mass * 1000
                             print(f"Warning: Invalid moment, using fallback")
                     except:
                         moment = mass * 1000
                     
                     body = pymunk.Body(mass, moment)
                     body.position = centroid
                     shape = pymunk.Poly(body, relative_points)
                     shape.elasticity = 0.6
                     shape.friction = 0.5
                     space.add(body, shape)
                     
                     active_shape.body = body
                     active_shape.poly = shape
            
            active_shape = None
            shape_offsets = [] # Clear offsets
        # PHYSICS STEP
        if physics_enabled:
            dt = 1.0 / 60.0
            space.step(dt)
            
        # Draw SHAPES (Physics Bodies)
        for shape in shapes:
            if shape.body and shape.poly:
                # Update points from physics body
                body = shape.body
                poly = shape.poly
                
                # Check if body position is valid (not NaN or infinite)
                import math
                if math.isnan(body.position.x) or math.isnan(body.position.y) or \
                   math.isinf(body.position.x) or math.isinf(body.position.y):
                    # Physics body has invalid position - remove it immediately
                    print(f"Warning: Shape has invalid physics position (NaN/Inf), removing from simulation")
                    try:
                        space.remove(body, poly)
                    except:
                        pass  # Already removed
                    shape.body = None
                    shape.poly = None
                    continue
                
                # Get world points
                world_points = []
                for v in poly.get_vertices():
                    # Rotate and translate
                    # v is pymunk.Vec2d, body.angle is in radians
                    p = v.rotated(body.angle) + body.position
                    
                    # Double-check for NaN in final position
                    if math.isnan(p.x) or math.isnan(p.y):
                        print(f"Warning: NaN detected in vertex position, skipping shape update")
                        world_points = shape.points  # Keep old points
                        break
                    
                    world_points.append((int(p.x), int(p.y)))
                
                shape.points = world_points
            
            # Draw shape
            # Draw shape
            if len(shape.points) > 1:
                draw_color = shape.color

                
                # Draw lines between points
                for i in range(len(shape.points)):
                    pt1 = shape.points[i]
                    pt2 = shape.points[(i + 1) % len(shape.points)]
                    cv2.line(img, pt1, pt2, draw_color, 3)
                    cv2.circle(img, pt1, 5, draw_color, cv2.FILLED)
                
                # Add measurements for first two adjacent sides only
                if len(shape.points) >= 4 and pixels_per_cm is not None and pixels_per_cm > 0:
                    import math
                    
                    # First side (0 to 1)
                    pt1 = shape.points[0]
                    pt2 = shape.points[1]
                    dist_px = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                    dist_cm = dist_px / pixels_per_cm
                    mid1 = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv2.putText(img, f"{dist_cm:.1f}cm", (mid1[0] - 30, mid1[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    # Second side (1 to 2)
                    pt1 = shape.points[1]
                    pt2 = shape.points[2]
                    dist_px = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                    dist_cm = dist_px / pixels_per_cm
                    mid2 = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv2.putText(img, f"{dist_cm:.1f}cm", (mid2[0] + 10, mid2[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw finalized lines
        for line in finalized_lines:
            point1, point2, distance_cm = line
            cv2.line(img, point1, point2, (255, 255, 255), 3)
            cv2.circle(img, point1, 8, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, point2, 8, (255, 255, 255), cv2.FILLED)
        
        # Draw Active Rotating Shape
        if rotation_mode and rotation_center is not None and active_shape:
            # Calculate rotated lines for display using OFFSETS and new CENTER (Thumb)
            rotated_shape_points = []
            for off in shape_offsets:
                 # Rotate offset in 2D (Z-axis only)
                 r_off = rotate_point((off[0], off[1]), (0, 0), active_shape.z_rotation)
                 # Add to new center
                 new_pt = (rotation_center[0] + r_off[0], rotation_center[1] + r_off[1])
                 rotated_shape_points.append(new_pt)
            
            # Draw rotated active shape
            if len(rotated_shape_points) > 1:
                for i in range(len(rotated_shape_points)):
                    pt1 = rotated_shape_points[i]
                    pt2 = rotated_shape_points[(i + 1) % len(rotated_shape_points)]
                    cv2.line(img, pt1, pt2, (200, 200, 255), 3)
                    cv2.circle(img, pt1, 5, (200, 200, 255), cv2.FILLED)
                
            # Draw rotation center (Thumb)
            cv2.circle(img, rotation_center, 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, rotation_center, 15, (0, 0, 255), 2)
            
        
        # Draw locked point 1 (START)
        if locked_point_1 is not None:
            cv2.circle(img, locked_point_1, 20, (0, 255, 0), 3)
            cv2.circle(img, locked_point_1, 5, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "START", (locked_point_1[0] - 35, locked_point_1[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw locked point 2 (END)
        if locked_point_2 is not None:
            # Apply straightening for preview
            straightened_point_2 = straighten_line(locked_point_1, locked_point_2) if locked_point_1 else locked_point_2
            
            cv2.circle(img, straightened_point_2, 20, (255, 0, 0), 3)
            cv2.circle(img, straightened_point_2, 5, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, "END", (straightened_point_2[0] - 25, straightened_point_2[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw current line between locked points (with straightening preview)
        if locked_point_1 is not None and locked_point_2 is not None:
            # Apply straightening
            straightened_point_2 = straighten_line(locked_point_1, locked_point_2)
            
            # Draw thick current line (YELLOW for active line)
            cv2.line(img, locked_point_1, straightened_point_2, (0, 255, 255), 4)
            
            # Calculate and display distance
            import math
            distance_pixels = math.sqrt((straightened_point_2[0] - locked_point_1[0])**2 + 
                                       (straightened_point_2[1] - locked_point_1[1])**2)
            mid_point = ((locked_point_1[0] + straightened_point_2[0]) // 2, 
                        (locked_point_1[1] + straightened_point_2[1]) // 2)
            
            # Convert to centimeters if calibrated
            if pixels_per_cm is not None and pixels_per_cm > 0:
                distance_cm = distance_pixels / pixels_per_cm
                cv2.putText(img, f"Distance: {distance_cm:.1f} cm", 
                           (mid_point[0] - 80, mid_point[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(img, f"Distance: {int(distance_pixels)}px", 
                           (mid_point[0] - 80, mid_point[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Display status information
        # Draw finalized lines
        for line in finalized_lines:
            point1, point2, distance_cm = line
            cv2.line(img, point1, point2, (255, 255, 255), 3)
            cv2.circle(img, point1, 8, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, point2, 8, (255, 255, 255), cv2.FILLED)
        
        # Draw Active Rotating Shape
        if rotation_mode and rotation_center is not None and active_shape:
            # Calculate rotated lines for display using OFFSETS and new CENTER (Thumb)
            rotated_shape_points = []
            for off in shape_offsets:
                 # Rotate offset in 2D (Z-axis only)
                 r_off = rotate_point((off[0], off[1]), (0, 0), active_shape.z_rotation)
                 # Add to new center
                 new_pt = (rotation_center[0] + r_off[0], rotation_center[1] + r_off[1])
                 rotated_shape_points.append(new_pt)
            
            # Draw rotated active shape
            if len(rotated_shape_points) > 1:
                for i in range(len(rotated_shape_points)):
                    pt1 = rotated_shape_points[i]
                    pt2 = rotated_shape_points[(i + 1) % len(rotated_shape_points)]
                    cv2.line(img, pt1, pt2, (200, 200, 255), 3)
                    cv2.circle(img, pt1, 5, (200, 200, 255), cv2.FILLED)
                
            # Draw rotation center (Thumb)
            cv2.circle(img, rotation_center, 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, rotation_center, 15, (0, 0, 255), 2)
            
        # Draw Mode Indicator Sidebar
        sidebar_color = (50, 50, 50)
        cv2.rectangle(img, (0, 0), (200, 100), sidebar_color, cv2.FILLED)
        
        mode_text = "DRAWING"
        text_color = (0, 255, 0) # Green
        if rotation_mode:
            mode_text = "ROTATION"
            text_color = (255, 0, 255) # Purple
            
        cv2.putText(img, "MODE:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(img, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # Draw locked point 1 (START)
        if locked_point_1 is not None:
            cv2.circle(img, locked_point_1, 20, (0, 255, 0), 3)
            cv2.circle(img, locked_point_1, 5, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "START", (locked_point_1[0] - 35, locked_point_1[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw locked point 2 (END)
        if locked_point_2 is not None:
            # Apply straightening for preview
            straightened_point_2 = straighten_line(locked_point_1, locked_point_2) if locked_point_1 else locked_point_2
            
            cv2.circle(img, straightened_point_2, 20, (255, 0, 0), 3)
            cv2.circle(img, straightened_point_2, 5, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, "END", (straightened_point_2[0] - 25, straightened_point_2[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw current line between locked points (with straightening preview)
        if locked_point_1 is not None and locked_point_2 is not None:
            # Apply straightening
            straightened_point_2 = straighten_line(locked_point_1, locked_point_2)
            
            # Draw thick current line (YELLOW for active line)
            cv2.line(img, locked_point_1, straightened_point_2, (0, 255, 255), 4)
            
            # Calculate and display distance
            import math
            distance_pixels = math.sqrt((straightened_point_2[0] - locked_point_1[0])**2 + 
                                       (straightened_point_2[1] - locked_point_1[1])**2)
            mid_point = ((locked_point_1[0] + straightened_point_2[0]) // 2, 
                        (locked_point_1[1] + straightened_point_2[1]) // 2)
            
            # Convert to centimeters if calibrated
            if pixels_per_cm is not None and pixels_per_cm > 0:
                distance_cm = distance_pixels / pixels_per_cm
                cv2.putText(img, f"Distance: {distance_cm:.1f} cm", 
                           (mid_point[0] - 80, mid_point[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(img, f"Distance: {int(distance_pixels)}px", 
                           (mid_point[0] - 80, mid_point[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Calculate and display FPS
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0:
            fps = 1 / time_diff
        else:
            fps = 0
        prev_time = curr_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        status_y = img.shape[0] - 60
        
        # Show locked points status
        status_text = f"Lines: {len(finalized_lines)} | "
        
        if rotation_mode:
            status_text += "ROTATION MODE - Left Thumb/Index to Rotate - Thumbs Up to Finish"
        elif locked_point_1 is not None and locked_point_2 is not None:
            status_text += "Line drawn - THUMBS UP to finalize"
        elif locked_point_1 is not None:
            status_text += "Point 1 LOCKED - Pinch Right for Point 2"
        elif locked_point_2 is not None:
            status_text += "Point 2 LOCKED - Pinch Left for Point 1"
        else:
            status_text += "Pinch to set points"
        
        cv2.putText(img, status_text, (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display instructions and calibration status
        instruction_text = "Thumbs Up: Finalize | 'U': Undo | 'C': Clear All | 'Q': Quit"
        if pixels_per_cm is not None:
            instruction_text += f" | Cal: {pixels_per_cm:.1f} px/cm"
        cv2.putText(img, instruction_text, 
                   (10, img.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Update thumbs up state for next frame
        thumbs_up_was_detected = thumbs_up_detected
        
        # Show the image
        cv2.imshow("Hand Gesture Drawing", img)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Exit on 'q' key
        if key == ord('q'):
            break
        # Undo last line on 'u' key
        elif key == ord('u'):
            if finalized_lines:
                finalized_lines.pop()
                print("Last line undone!")
            else:
                print("No lines to undo")
        # Clear locked points on 'c' key
        elif key == ord('c'):
            locked_point_1 = None
            locked_point_2 = None
            finalized_lines = []
            
            # Remove all shapes from physics space
            for shape in shapes:
                if shape.body:
                    space.remove(shape.body, shape.poly)
            shapes = []
            
            print("All lines and shapes cleared!")
        
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
