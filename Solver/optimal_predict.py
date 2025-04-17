import numpy as np
from .optimal_simulator import BilliardsPhysicsEngine, Turn, PhysicsConfig

def calculate_optimal_hitting_angle(cue_pos, target_pos, target_radius):
    """
    큐구가 타겟 공을 정확히 맞추기 위한 최적의 각도를 계산
    
    타겟 공의 좌우를 맞추는 두 가지 각도를 반환
    큐구가 타겟 공의 중심을 향해 치면 타겟 공이 똑바로 진행하지만,
    약간 비스듬히 치면 타겟 공이 각도를 가지고 진행
    """
    dx = target_pos[0] - cue_pos[0]
    dy = target_pos[1] - cue_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # 타겟 공의 중심을 향하는 직선 각도
    center_angle = np.arctan2(dy, dx)
    
    # 타겟 공을 스쳐 지나가는 각도 계산 (사인 법칙 사용)
    if distance == 0:
        return []
    
    # 두 공의 반지름 합에 대한 역사인
    sin_theta = (2 * target_radius) / distance
    if sin_theta > 1:  # 공들이 너무 가까워 충돌이 불가능한 경우
        return []
        
    # 타겟 공을 맞출 수 있는 두 가지 각도 계산
    theta = np.arcsin(sin_theta)
    return [
        center_angle - theta,  # 왼쪽으로 비스듬히 치는 각도
        center_angle + theta   # 오른쪽으로 비스듬히 치는 각도
    ]

def is_first_collision_red(collision_history):
        """첫 번째 충돌이 빨간 공과의 충돌인지 확인"""
        if not collision_history:
            return False
        first_collision = collision_history[0]
        return (('yellow', 'red1') == first_collision or 
                ('yellow', 'red2') == first_collision or
                ('red1', 'yellow') == first_collision or 
                ('red2', 'yellow') == first_collision)

def find_optimal_angle(positions):
    physics_engine = BilliardsPhysicsEngine()
    physics_engine.set_positions(positions)
    
    # 현재 차례의 공과 타겟 공들의 위치 확인
    cue_ball = None
    target_balls = []
    
    for ball in physics_engine.balls:
        if ball['color'] == Turn.YELLOW.value:
            cue_ball = ball
        elif ball['color'] in ['red1', 'red2']:
            target_balls.append(ball)
    
    # 각 타겟 공에 대해 가능한 히팅 각도 계산
    best_angles = []
    for target_ball in target_balls:
        hitting_angles = calculate_optimal_hitting_angle(
            cue_ball['position'],
            target_ball['position'],
            physics_engine.ball_radius
        )
        best_angles.extend(hitting_angles)
    
    # 직접 히팅이 불가능한 경우를 위한 추가 각도들
    if not best_angles:
        best_angles = np.linspace(0, 2 * np.pi, 16)  # 360도를 16등분
    
    # 각 후보 각도에 대해 좁은 범위의 각도를 테스트
    angle_variations = np.linspace(-0.1, 0.1, 5)
    
    best_result = None
    best_angle = 0  # 기본값 설정
    best_first_collision = False  # 첫 충돌이 빨간 공인지 여부
    min_collisions = float('inf')
    
    for base_angle in best_angles:
        for variation in angle_variations:
            test_angle = base_angle + variation
            
            # 시뮬레이션 실행
            physics_engine.set_positions(positions)
            physics_engine.simulate(test_angle, PhysicsConfig.DEFAULT_POWER, Turn.YELLOW)
            
            # 성공한 경우의 처리
            if physics_engine.result == 'success':
                collision_count = len(physics_engine.collision_history)
                first_red = is_first_collision_red(physics_engine.collision_history)
                
                # 첫 충돌이 빨간 공인 경우를 우선적으로 선택
                if first_red and not best_first_collision:
                    best_first_collision = True
                    min_collisions = collision_count
                    best_result = physics_engine.result
                    best_angle = test_angle
                # 이미 첫 충돌이 빨간 공인 경우들 중에서는 충돌 횟수가 적은 것을 선택
                elif first_red and best_first_collision and collision_count < min_collisions:
                    min_collisions = collision_count
                    best_result = physics_engine.result
                    best_angle = test_angle
                # 아직 성공 케이스를 못 찾은 경우에는 일단 저장
                elif not best_first_collision and best_result != 'success':
                    min_collisions = collision_count
                    best_result = physics_engine.result
                    best_angle = test_angle
    
	# 직접적인 히팅이 실패한 경우, 쿠션을 활용한 샷을 시도
    if best_result != 'success':
        cushion_angles = np.linspace(0, 2 * np.pi, 32)  # 더 많은 각도 시도
        for angle in cushion_angles:
            physics_engine.set_positions(positions)
            physics_engine.simulate(angle, PhysicsConfig.DEFAULT_POWER, Turn.YELLOW)
            
            if physics_engine.result == 'success':
                collision_count = len(physics_engine.collision_history)
                first_red = is_first_collision_red(physics_engine.collision_history)
                
                if first_red and not best_first_collision:
                    best_first_collision = True
                    min_collisions = collision_count
                    best_result = physics_engine.result
                    best_angle = angle
                elif first_red and best_first_collision and collision_count < min_collisions:
                    min_collisions = collision_count
                    best_result = physics_engine.result
                    best_angle = angle
                elif not best_first_collision and best_result != 'success':
                    min_collisions = collision_count
                    best_result = physics_engine.result
                    best_angle = angle
    
    # 최종 결과 처리
    if best_result != 'success':
        print("Not Found.")
        # 기본 각도로 첫 번째 빨간 공을 향해 치기
        default_angle = np.arctan2(
            target_balls[0]['position'][1] - cue_ball['position'][1],
            target_balls[0]['position'][0] - cue_ball['position'][0]
        )
        best_angle = default_angle
        
    return best_angle  # 항상 유효한 각도를 반환