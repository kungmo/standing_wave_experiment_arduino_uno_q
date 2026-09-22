// 28BYJ-48 스텝모터용 두꺼운 실 도르래
// 외경 40 mm, 전체 높이 18 mm
// 얇은 테두리 + 깊고 가파른 중앙 홈

pulley_z = 18;
pulley_r = 20;           // 도르래 지름: 40 mm

hub_a = 3.2;             // 28BYJ-48 축의 평평한 면 방향
hub_b = 5;               // 축 구멍의 다른 방향

groove_depth = 10;       // 깊은 허리: 홈 바닥 반지름 10 mm
groove_bottom_width = 3; // 두꺼운 실이 닿는 바닥 폭
rim_width = 1.2;         // 얇은 위·아래 테두리

screw_r = 2;             // M4 고정 나사 구멍 반지름
screw_z = pulley_z / 2;

module body() {
    difference() {
        // 축 방향 돌출부가 없는 평탄한 외형
        cylinder(h = pulley_z, r = pulley_r);

        // 28BYJ-48 축 결합용 구멍
        translate([-hub_a / 2, -hub_b / 2, 0])
            cube([hub_a, hub_b, pulley_z], center = false);
    }
}

module screws() {
    // 도르래 측면에서 축 중심을 향하는 M4 고정 나사 구멍
    translate([0, 0, screw_z])
        rotate(90, [0, 1, 0])
            translate([0, 0, -pulley_r - 1])
                cylinder(h = 2 * pulley_r + 2, r = screw_r);
}

module groove() {
    bottom_z1 = (pulley_z - groove_bottom_width) / 2;
    bottom_z2 = (pulley_z + groove_bottom_width) / 2;

    // 얇은 바깥 턱과 깊고 가파른 경사면
    rotate_extrude(convexity = 100)
        polygon([
            [pulley_r, rim_width],
            [pulley_r - groove_depth, bottom_z1],
            [pulley_r - groove_depth, bottom_z2],
            [pulley_r, pulley_z - rim_width]
        ]);
}

difference() {
    body();
    groove();
    screws();
}