class SpatialGrid:
    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size
        self.cols = int(width / cell_size) + 1
        self.rows = int(height / cell_size) + 1
        self.grid = {}  # Dictionary to store balls in each cell
        
    def get_cell_index(self, position):
        """주어진 위치의 그리드 셀 인덱스를 반환"""
        col = int(position[0] / self.cell_size)
        row = int(position[1] / self.cell_size)
        return (col, row)
    
    def get_neighbor_cells(self, cell):
        """주어진 셀의 이웃 셀들을 반환 (자신 포함)"""
        col, row = cell
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_col = col + i
                new_row = row + j
                if 0 <= new_col < self.cols and 0 <= new_row < self.rows:
                    neighbors.append((new_col, new_row))
        return neighbors

    def update(self, balls):
        """볼들의 위치를 기반으로 그리드 업데이트"""
        self.grid.clear()
        for i, ball in enumerate(balls):
            cell = self.get_cell_index(ball['position'])
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(i)
            
    def get_potential_collisions(self):
        """충돌 가능성이 있는 볼 쌍들을 반환"""
        potential_pairs = set()
        for cell in self.grid:
            # 현재 셀 내의 볼들 사이의 충돌 검사
            balls_in_cell = self.grid[cell]
            for i in range(len(balls_in_cell)):
                for j in range(i + 1, len(balls_in_cell)):
                    potential_pairs.add((min(balls_in_cell[i], balls_in_cell[j]), 
                                      max(balls_in_cell[i], balls_in_cell[j])))
            
            # 이웃 셀들과의 충돌 검사
            for neighbor in self.get_neighbor_cells(cell):
                if neighbor in self.grid and neighbor != cell:
                    for ball1 in self.grid[cell]:
                        for ball2 in self.grid[neighbor]:
                            potential_pairs.add((min(ball1, ball2), max(ball1, ball2)))
                            
        return list(potential_pairs)