import numpy as np
from enum import Enum
from .grid import SpatialGrid

class BallConfig:
    RADIUS = 15.0/2  # 당구공 반지름
    MASS = 57.0     # 당구공 질량
    COLORS = {
        'white': 'white',   # 흰 공
        'yellow': 'yellow', # 노란 공
        'red1': 'red1',     # 빨간 공 1
        'red2': 'red2'      # 빨간 공 2
    }

class TableConfig:
    WIDTH = 600   # 당구대 가로 길이
    HEIGHT = 300   # 당구대 세로 길이

class PhysicsConfig:
    FRICTION_COEF = 0.99    # 마찰 계수 증가
    RESTITUTION_COEF = 0.8  # 반발 계수 감소
    MIN_VELOCITY = 1.0      # 최소 속도 증가
    DEFAULT_POWER = 30.0    # 기본 파워 설정
    MAX_POWER = 500.0      # 최대 파워 설정

class Turn(Enum):
    WHITE = 'white'
    YELLOW = 'yellow'

class BilliardsPhysicsEngine:
    def __init__(self):
        self.table_width = TableConfig.WIDTH
        self.table_height = TableConfig.HEIGHT
        self.friction_coef = PhysicsConfig.FRICTION_COEF
        self.restitution_coef = PhysicsConfig.RESTITUTION_COEF
        self.ball_radius = BallConfig.RADIUS
        self.current_turn = Turn.YELLOW
        
        # 공간 분할을 위한 그리드 초기화
        self.spatial_grid = SpatialGrid(self.table_width, self.table_height, cell_size=self.ball_radius * 4)
        
        self.balls = []
        self.history = []
        self.collision_history = []
        self.collision_history_with_ball = []
        self.collision_history_with_wall = []
        
        self.result = None
        self.success_angles = []
        self.success_history = []
        self.angle = 0.0

    def set_positions(self, positions):
        """공들의 초기 위치를 설정"""
        self.balls = []
        colors = list(BallConfig.COLORS.values())

        for (x, y), color in zip(positions, colors):
            x = float(min(max(x, self.ball_radius), self.table_width - self.ball_radius))
            y = float(min(max(y, self.ball_radius), self.table_height - self.ball_radius))
        
            self.balls.append({
                'position': np.array([x, y]),
                'velocity': np.array([0.0, 0.0]),
                'color': color,
                'mass': BallConfig.MASS
            })
    
        self.history = [[ball['position'].copy()] for ball in self.balls]
        self.collision_history.clear()
        self.collision_history_with_ball.clear()
        self.collision_history_with_wall.clear()

    def simulate(self, angle, power, turn):
        """최적화된 시뮬레이션 실행"""
        # 초기화
        self.current_turn = turn
        self.angle = angle
        self.collision_history_with_ball = []
        self.collision_history_with_wall = []
        self.collision_history = []
        self.result = None
        
        # 현재 차례의 공 찾기
        current_ball = None
        for ball in self.balls:
            if ball['color'] == turn.value:
                current_ball = ball
                break

        # 초기 속도 설정
        dt = 0.1
        velocity = np.array([
            np.cos(angle) * power / dt,
            np.sin(angle) * power / dt
        ])
        current_ball['velocity'] = velocity
        
        # 시뮬레이션 제한 설정
        MAX_FRAMES = 1000
        frame_count = 0
        
        all_stopped = False
        while not all_stopped and frame_count < MAX_FRAMES:
            frame_count += 1
            all_stopped = True
            
            # 공들의 위치와 속도 업데이트
            for i, ball in enumerate(self.balls):
                # 마찰에 의한 감속
                ball['velocity'] *= self.friction_coef

                # 최소 속도 이하면 정지
                if np.all(np.abs(ball['velocity']) < PhysicsConfig.MIN_VELOCITY):
                    ball['velocity'] = np.array([0.0, 0.0])
                else:
                    all_stopped = False

                # 위치 업데이트
                ball['position'] += ball['velocity'] * dt
                self.history[i].append(ball['position'].copy())
                
                # 벽과의 충돌 처리
                self._handle_wall_collision(ball)
            
            # 공들 사이의 충돌 처리
            self._handle_ball_collisions()

        # 결과 판정
        if ((turn.value, 'red1') in self.collision_history and (turn.value, 'red2') in self.collision_history) or \
           ((turn.value, 'red1') in self.collision_history and ('red2', turn.value) in self.collision_history) or \
           (('red1', turn.value) in self.collision_history and ('red2', turn.value) in self.collision_history) or \
           (('red1', turn.value) in self.collision_history and (turn.value, 'red2') in self.collision_history):
            self.result = 'success'
            self.success_angles.append(angle)
            self.success_history.append(self.collision_history.copy())

        elif ((turn.value, 'red1') in self.collision_history or \
              (turn.value, 'red2') in self.collision_history or \
              ('red1', turn.value) in self.collision_history or \
              ('red2', turn.value) in self.collision_history):
            self.result = 'partial'

        elif self.collision_history_with_ball == []:
            self.result = 'fail'

        if ('yellow', 'white') in self.collision_history or ('white', 'yellow') in self.collision_history:
            self.result = 'foul'

    def _handle_wall_collision(self, ball):
        """벽과의 충돌 처리"""
        margin = self.ball_radius

        # x축 벽과의 충돌
        if ball['position'][0] < margin:
            if ball['color'] == self.current_turn.value:
                self.collision_history_with_wall.append((self.current_turn.value, 'wall'))
                self.collision_history.append((self.current_turn.value, 'wall'))
            ball['position'][0] = margin
            ball['velocity'][0] *= -self.restitution_coef
        elif ball['position'][0] > self.table_width - margin:
            if ball['color'] == self.current_turn.value:
                self.collision_history_with_wall.append((self.current_turn.value, 'wall'))
                self.collision_history.append((self.current_turn.value, 'wall'))
            ball['position'][0] = self.table_width - margin
            ball['velocity'][0] *= -self.restitution_coef

        # y축 벽과의 충돌
        if ball['position'][1] < margin:
            if ball['color'] == self.current_turn.value:
                self.collision_history_with_wall.append((self.current_turn.value, 'wall'))
                self.collision_history.append((self.current_turn.value, 'wall'))
            ball['position'][1] = margin
            ball['velocity'][1] *= -self.restitution_coef
        elif ball['position'][1] > self.table_height - margin:
            if ball['color'] == self.current_turn.value:
                self.collision_history_with_wall.append((self.current_turn.value, 'wall'))
                self.collision_history.append((self.current_turn.value, 'wall'))
            ball['position'][1] = self.table_height - margin
            ball['velocity'][1] *= -self.restitution_coef

    def _handle_ball_collisions(self):
        """최적화된 공 충돌 처리"""
        # 그리드 업데이트
        self.spatial_grid.update(self.balls)
        potential_pairs = self.spatial_grid.get_potential_collisions()
        
        for i, j in potential_pairs:
            ball1, ball2 = self.balls[i], self.balls[j]
            diff = ball2['position'] - ball1['position']
            distance = np.linalg.norm(diff)
            
            if distance < 2 * self.ball_radius * 1.01:
                collision = (ball1['color'], ball2['color'])
                self.collision_history_with_ball.append(collision)
                self.collision_history.append(collision)

                normal = diff / distance
                vel1_norm = np.linalg.norm(ball1['velocity'])
                dir1 = ball1['velocity'] / vel1_norm if vel1_norm > 0 else np.array([0.0, 0.0])
                impact_angle = np.dot(normal, dir1)

                overlap = (2 * self.ball_radius - distance) / 2
                ball1['position'] -= overlap * normal
                ball2['position'] += overlap * normal

                relative_velocity = ball1['velocity'] - ball2['velocity']

                if abs(impact_angle) > 0.9:
                    reduced_mass = (ball1['mass'] * ball2['mass']) / (ball1['mass'] + ball2['mass'])
                    impulse = 2 * reduced_mass * np.dot(relative_velocity, normal) * normal
                    ball1['velocity'] -= (impulse / ball1['mass']) * self.restitution_coef
                    ball2['velocity'] += (impulse / ball2['mass']) * self.restitution_coef
                else:
                    tangent = np.array([-normal[1], normal[0]])
                    normal_vel1 = np.dot(ball1['velocity'], normal) * normal
                    normal_vel2 = np.dot(ball2['velocity'], normal) * normal
                    tangent_vel1 = np.dot(ball1['velocity'], tangent) * tangent
                    tangent_vel2 = np.dot(ball2['velocity'], tangent) * tangent
                    ball1['velocity'] = (normal_vel2 + tangent_vel1) * self.restitution_coef
                    ball2['velocity'] = (normal_vel1 + tangent_vel2) * self.restitution_coef