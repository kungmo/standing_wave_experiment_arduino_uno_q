
hub_z = 7;
hub_r = 8;
hub_a = 3.2;
hub_b = 5;

wheel_z = 14;
wheel_r = 20;

v_depth = 10;

module body() {
 carve_x = 10;
 total_z = hub_z + wheel_z;
 difference() {
   cylinder(h=total_z,r=hub_r);
   translate([-hub_a/2,-hub_b/2,0]) {
     cube([hub_a,hub_b,total_z], 1);
   }
 }
 cylinder(h=wheel_z,r=wheel_r);
}

module screws() {
  translate([0,0,wheel_z+(hub_z/2)]) 
    rotate(90,[0,1,0]) 
      translate([0,0,-15]) 
        cylinder(h=30,r=2);
}

module groove() {
 rotate_extrude(convexity = 100) 
   translate([24.25-3, 0, 0])
     rotate(45)
       square([v_depth,v_depth],0); 
}

module no_screws() {
  difference() {
    body();
    groove();
  }
}

difference() {
  no_screws();
  screws();
}