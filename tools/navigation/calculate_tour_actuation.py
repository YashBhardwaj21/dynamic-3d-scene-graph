import json, math, sys
from pathlib import Path

FUSED_DIR = Path('outputs/semantic_mapping_fused_desk_run')
sys.path.append('.')
from nav import FastAStarPlanner, TARGET_DATABASE, normalize_angle, SPEED_MPS, TURN_RADPS

planner = FastAStarPlanner(FUSED_DIR)

tour_steps = [
    ('desk_001', 'desk 1', 'Primary Workstation Desk'),
    ('chair_002', 'chair 2', 'Conference Side Chair'),
    ('chair_003', 'chair 3', 'Central Meeting Chair'),
    ('box_001', 'box 1', 'Storage Box by Desk 1'),
    ('chair_005', 'chair 5', 'South Meeting Chair'),
    ('desk_005', 'desk 5', 'South Laboratory Bench'),
    ('desk_018', 'desk 18', 'West Workstation Bench'),
    ('desk_017', 'desk 17', 'Northwest Center Bench'),
    ('desk_023', 'desk 23', 'West Corridor Workstation'),
    ('desk_024', 'desk 24', 'North Corridor Desk')
]

cur_x, cur_y, cur_yaw = 0.0, 0.0, 0.0
total_tour_dist = 0.0
total_tour_drive_time = 0.0
total_tour_turn_time = 0.0

results = []

for idx, (tid, cmd, desc) in enumerate(tour_steps, 1):
    info = TARGET_DATABASE[tid]
    tgt_x, tgt_y, tgt_yaw = info['approach']
    waypoints = planner.plan((cur_x, cur_y), (tgt_x, tgt_y))
    
    seg_maneuvers = []
    heading = cur_yaw
    step_dist = 0.0
    step_drive_time = 0.0
    step_turn_time = 0.0
    
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i+1]
        dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        bearing = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
        turn = normalize_angle(bearing - heading)
        
        turn_t = abs(turn) / TURN_RADPS if abs(turn) > 0.05 else 0.0
        drive_t = dist / SPEED_MPS
        
        step_dist += dist
        step_drive_time += drive_t
        step_turn_time += turn_t
        
        seg_maneuvers.append({
            'from': [round(p1[0], 2), round(p1[1], 2)],
            'to': [round(p2[0], 2), round(p2[1], 2)],
            'dist_m': round(dist, 2),
            'bearing_deg': round(math.degrees(bearing), 1),
            'turn_deg': round(math.degrees(turn), 1),
            'turn_s': round(turn_t, 2),
            'drive_s': round(drive_t, 2),
        })
        heading = bearing
        
    final_turn = normalize_angle(tgt_yaw - heading)
    final_turn_t = abs(final_turn) / TURN_RADPS if abs(final_turn) > 0.05 else 0.0
    step_turn_time += final_turn_t
    
    total_tour_dist += step_dist
    total_tour_drive_time += step_drive_time
    total_tour_turn_time += step_turn_time
    
    results.append({
        'step': idx,
        'id': tid,
        'cmd': f'python nav.py "{cmd}"',
        'desc': desc,
        'start_pose': [round(cur_x, 2), round(cur_y, 2), round(math.degrees(cur_yaw), 1)],
        'target_pose': [round(tgt_x, 2), round(tgt_y, 2), round(math.degrees(tgt_yaw), 1)],
        'dist_m': round(step_dist, 2),
        'drive_s': round(step_drive_time, 2),
        'turn_s': round(step_turn_time, 2),
        'transit_s': round(step_drive_time + step_turn_time, 2),
        'waypoints': len(waypoints),
        'final_turn_deg': round(math.degrees(final_turn), 1),
        'final_turn_s': round(final_turn_t, 2),
        'segments': seg_maneuvers
    })
    
    cur_x, cur_y, cur_yaw = tgt_x, tgt_y, tgt_yaw

summary = {
    'total_dist_m': round(total_tour_dist, 2),
    'total_drive_s': round(total_tour_drive_time, 2),
    'total_turn_s': round(total_tour_turn_time, 2),
    'total_transit_s': round(total_tour_drive_time + total_tour_turn_time, 2),
    'steps': results
}

with open('outputs/semantic_mapping_fused_desk_run/visualizations/10_observations/trajectory_actuation_telemetry.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("SUCCESS: Actuation calculations complete.")
print(f"Total distance: {summary['total_dist_m']} m | Total drive: {summary['total_drive_s']} s | Total turn: {summary['total_turn_s']} s | Total transit: {summary['total_transit_s']} s")
