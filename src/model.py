# pyrefly: ignore [missing-import]
#3 câu hỏi phỏng vấn tổng kết Phase 1 — tự trả lời để check hiểu:
#1. Tại sao class VietPopNet phải kế thừa nn.Module, và tại sao bắt buộc gọi super().__init__()?
#2. Sự khác nhau giữa __init__ và forward là gì?
#3. Vì sao dùng nn.ModuleList thay vì list Python thường khi có nhiều layer?
#hidden_dims = linear + ReLU 
#dims = in_features + hidden_dims -> từng cặp [in_features, out_features] một

import torch.nn as nn 
import torch
class VietPopNet(nn.Module):
    def __init__(self, in_features, hidden_dims):
        super().__init__()
        self.hidden_layer = nn.ModuleList() 
        dims = [in_features] + hidden_dims 
        for i in range(len(dims) - 1): 
            self.hidden_layer.append(nn.Linear(dims[i], dims[i+1]))
        self.output_layer = nn.Linear(dims[-1], 1)

    def forward(self, x): 
        for layer in self.hidden_layer:
            x = layer(x) 
            x = torch.relu(x) 
        
        x = self.output_layer(x) 
        return x 
    
if __name__ == '__main__': 
    model = VietPopNet(in_features=20, hidden_dims=[128,64])
    print(model) 

    dummy_x = torch.randn(8,20) 
    out = model(dummy_x)
    print("Số chiều vào:", dummy_x.shape)
    print("Số chiều ra:", out.shape) 


    
