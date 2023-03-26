import numpy as np
#import dataclasses

#@dataclasses.dataclass
class Bbox:
    def __init__(self, x_min:int,
                    y_min:int,
                    x_max:int,
                    y_max:int) -> None:    
        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)
        
    def array(self):
        return [self.x_min, self.y_min, self.x_max, self.y_max]
    
    def get_h_w(self, size ):
        mask = np.zeros((size[0],size[1]),dtype=np.uint8) # initialize mask
        mask[self.y_min:self.y_max,self.x_min:self.x_max] = 255
        return mask
        
    def get_p1_p2(self):
        return (self.x_min,self.y_min),(self.x_max,self.y_max)
    
    def __repr__(self) -> str:
        return f'x1:{self.x_min},y1:{self.y_min},x2:{self.x_max},y2:{self.y_max}\n'
    
    def __str__(self) -> str:
        return f'x1:{self.x_min},y1:{self.y_min},x2:{self.x_max},y2:{self.y_max}'