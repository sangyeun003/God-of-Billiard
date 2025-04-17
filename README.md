# God of Billiard
## Project Overview
서울시립대학교 지능형로봇 팀 프로젝트

![Demo](./report/demo.gif)

Task : 4구 당구 경기에서 최적의 샷 방향을 계산하여 시각화하는 지능형 로봇 시스템 구현

Project Period : 2024년 11월 ~ 2024년 12월

Tech : Python, OpenCV

Member : 2명

Project Wrap Up Report : [click here](./report/지능형로봇_결과보고서.pdf)

Project Slides : [click here](./report/IR_project_presentation.pptx)

Environments : Nvidia Jetson Javier NX, Logitech Web Camera

### Pipeline
1. 당구대 & 당구공 Detection
	- 이미지 좌표계(2D pixel coordinate)에서 당구대의 꼭짓점과 당구공 중심 좌표 계산
2. 당구대의 꼭짓점과 당구공의 좌표를 당구대 기준 좌표계(3D world coordinate)로 변환
	- 당구대 평면이 ```Z=0```
3. 변환한 좌표를 이용하여 칠 방향 계산(3D world coordinate)
4. 칠 방향 위의 한 점을 2D pixel coordinate로 변환 후, 화살표로 시각화
5. 1, 2, 4, 5번 반복. ```c```를 누르는 경우 3번도 재실행

## Content
1. [Detection Part](#Detection-Part)
2. [Solver Part](#Solver-Part)

## Detection Part
1. Table detection
2. Ball detection
3. Coordinate transformation (2D image coordinate -> 3D world coordinate)

## Solver Part
1. Get ball information with 3D coordinate
2. Simulation
3. Return optimal angle

## Reference
- https://github.com/Detail-AR/Detail_AR
- Zhengyou Zhang. A Flexible New Technique for Camera Calibration. T-PAMI 2000