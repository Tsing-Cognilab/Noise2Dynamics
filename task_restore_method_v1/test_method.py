"""Small contract checks; usefulness remains a real closed-loop question."""
import unittest
from common import *


class MethodTests(unittest.TestCase):
    def test_causal(self):
        torch.manual_seed(3)
        model = Restorer(width=8).eval()
        nn.init.normal_(model.out.weight,std=.01)
        x = torch.rand(2,2,3,84,84)
        y = x.clone()
        y[:,1] = torch.rand_like(y[:,1])
        with torch.no_grad():
            first = model(x)[:,0]
            second = model(y)[:,0]
        self.assertTrue(torch.equal(first,second))

    def test_identity_start_and_gradient(self):
        model = Restorer(width=8)
        x = torch.rand(2,2,3,84,84).clamp(.01,.99)
        self.assertLess(float((model(x)-x).abs().max()),1e-6)
        model(x).mean().backward()
        self.assertGreater(float(model.out.weight.grad.abs().sum()),0.)

    def test_exposure_linear_not_srgb(self):
        image = torch.tensor([.2,.5,.8]).reshape(1,1,3,1,1)
        linear = torch.where(image<=.04045,image/12.92,((image+.055)/1.055)**2.4)*.25
        dark = torch.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055)
        self.assertLess(float((exposure_compensate(dark)-image).abs().max()),1e-6)

    def test_matching_noise_rng(self):
        clean = torch.rand(4,2,2,3,12,12)
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        self.assertTrue(torch.equal(noise_tensor(clean,g1),noise_tensor(clean,g2)))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
