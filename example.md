# TProductDetail Schema

```typescript
export type TProductDetail = {
  id: number
  code: string
  name: string
  description: string
  brand: Brand
  is_active: boolean
  category: Category
  rating: { score: number; total: number; breakdown: Record<string, number> }
  skus: { id: number; price: number; original_price: number; attributes: { name: string; value: string }[]; sold: number }[]
  specifications: Record<string, string>
}
```

## Example

```json
{"id":1,"code":"camisa-denim-top-retro-mujer-estilo-hong-kong-tie-tencel-principios-otono-slim-fit-nuevo-pequena.afa978f3-27e7-495c-a515-a93a951ff965","name":"Camisa Denim Top Retro Mujer Estilo Hong Kong Tie Tencel Principios Otoño Slim-Fit Nuevo Pequeña","description":"Consejos\nAcerca de la aberración cromática: colóquese...","brand":{"id":1,"code":"unknown","name":"Unknown","description":"Unknown"},"is_active":true,"category":{"id":1,"name":"Camisas y Blusas","description":"Camisas y Blusas","parent_id":null},"rating":{"score":2.5,"total":1,"breakdown":{"5":0,"4":0,"3":0,"2":1,"1":0}},"skus":[{"id":1,"price":868,"original_price":868,"attributes":[{"name":"Color","value":"Azul vaquero nostálgico"},{"name":"Tamaño","value":"2XL"}],"sold":0}],"specifications":{}}
```
