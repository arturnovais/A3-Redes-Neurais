# A3-Redes-Neurais

Repositório dedicado ao terceiro trabalho da matéria de Redes Neurais. O projeto visa explorar diferentes arquiteturas, utilizando camadas convolucionais, ligações residuais e camadas de normalização.

## Arquiteturas Implementadas

1. **Arquitetura Linear**
   - Perceptron sem função de ativação.
   
2. **MLP (Multi-Layer Perceptron)**
   - Rede com até 20 mil parâmetros.
   
3. **Arquitetura Convolucional LeNet**
   - Utilizando menos parâmetros que a MLP.
   
4. **Arquitetura Convolucional VGG**
   
5. **Arquitetura Convolucional ResNet**

### Implementação Extra
- Rede recorrente.
- Rede estilo ViT (presença de um mecanismo de atenção).

## Objetivos da Comparação

1. **Modelo Linear vs. MLP**
   - Demonstrar que aumentar o número de camadas (adicionar não linearidade) melhora o resultado.
   
2. **MLP vs. Arquitetura LeNet**
   - Mostrar que a LeNet, mesmo com menos parâmetros, consegue um resultado melhor, demonstrando o "poder" da convolução.
   
3. **Implementação da VGG com Diferentes Tamanhos de Bloco**
   - Testar e discutir os resultados.
   
4. **VGG com x Blocos vs. VGG com 2x Blocos**
   - Demonstrar que uma VGG com 2x blocos performa pior que uma VGG com x blocos.
   - Mostrar que isso muda se utilizarmos ligações residuais.
   - Reproduzir os resultados obtidos no artigo das residuais.
