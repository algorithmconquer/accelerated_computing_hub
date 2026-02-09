 step1:nsys profile --force-overwrite true -o power_iteration__baseline python 06__asynchrony__power_iteration_solved.py
 step2:nsys stats power_iteration__baseline.nsys-rep


 
 ** OS Runtime Summary (osrt_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)  Min (ns)   Max (ns)    StdDev (ns)     Name   
 --------  ---------------  ---------  -----------  --------  --------  -----------  ------------  ---------
    100.0    4,069,850,088        910  4,472,362.7   1,903.0       861  464,571,657  22,111,043.2  fopen64  
      0.0          214,947         35      6,141.3   3,256.0       531       40,507       8,528.5  fread    
      0.0          154,987         58      2,672.2   2,204.5       761        9,979       1,784.6  fclose   
      0.0           46,931          6      7,821.8   6,542.5     3,928       12,945       3,566.1  mmap64   
      0.0           46,325         89        520.5     371.0       360       11,011       1,131.1  fputs    
      0.0           17,649         68        259.5     180.0        40        3,236         387.5  sigaction
      0.0           11,933          1     11,933.0  11,933.0    11,933       11,933           0.0  munmap   
      0.0              770         11         70.0      50.0        40          320          83.1  flockfile
      0.0              571          1        571.0     571.0       571          571           0.0  fwrite   
      0.0              512          5        102.4      90.0        50          231          74.6  fflush   


      从 nsys stats 的输出结果可以明确判断：您的性能分析文件中没有捕获到任何 CUDA 相关数据（Kernel、内存传输、API 调用等），仅有操作系统运行时（OS Runtime）的文件 I/O 操作记录。这说明程序在运行过程中未实际触发 GPU 计算。




