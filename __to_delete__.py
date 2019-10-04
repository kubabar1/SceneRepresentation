import torch
from torch.distributions import MultivariateNormal
from torch.distributions import Normal
from torch.distributions import Independent
import torch.distributions.kl
from torchvision import transforms, datasets

loc = torch.zeros(2)
scale = torch.ones(2)
mvn = MultivariateNormal(loc, torch.diag(scale))
[mvn.batch_shape, mvn.event_shape]
[torch.Size(()), torch.Size((3,))]
normal = Normal(loc, scale)
[normal.batch_shape, normal.event_shape]
[torch.Size((3,)), torch.Size(())]
diagn = Independent(normal, 1)
[diagn.batch_shape, diagn.event_shape]

n1 = torch.distributions.Normal(torch.tensor([0., 1., 2., 3.]), torch.tensor([2., 4., 6., 8.]))
n2 = torch.distributions.Normal(torch.tensor([0., 1.5, 2.5, 2.5]), torch.tensor([1.5, 4.5, 5.5, 7.7]))

print(n1)
print(n2)

print(n1.sample())
print(n2.sample())
print()

out = torch.distributions.kl.kl_divergence(n1, n2)
print(out)

a = torch.tensor([3, 2, 4])
b = torch.tensor([2, 1, 2])

print(b-a)


x = torch.randn(1, 1, 1, 64)
print(x.size())
#x.unsqueeze_(-1)
print(x.expand(1, 256, 64, 64).size())


transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

images_set = datasets.ImageFolder(root="data", transform=transform)

print(images_set)