# Brand artwork

Vector sources for the artwork that is not part of the app's component set.

| File                  | Used by                                                             |
| --------------------- | ------------------------------------------------------------------- |
| `opengraph-image.svg` | Source for `app/opengraph-image.png`, the social/link preview card. |
| `architecture.svg`    | Embedded directly in `README.md`. No raster version is generated.   |

The product logo set lives in `public/` instead, because the app serves those
files directly.

## Regenerating the OG card

`app/opengraph-image.png` is committed as a raster because Next.js serves it
through the `opengraph-image` file convention, and because rendering text to an
image at request time would make the card depend on fonts being installed on
the server. Regenerate it after editing the SVG:

```bash
node -e "require('sharp')('docs/brand/opengraph-image.svg',{density:144})\
  .resize(1200,630).png({compressionLevel:9})\
  .toFile('app/opengraph-image.png')"
```

`sharp` is not a declared dependency — it resolves transitively through Next.js
— so this is a one-off maintenance command rather than an `npm run` script. If
it is ever missing, `npm i -D sharp` for the regeneration and remove it again.

Both files use the fixed brand teal `#155E69` and reuse the mark geometry from
`public/logo-mark.svg` verbatim, so the lockup cannot drift out of sync. Neither
uses a gradient: `app/globals.css` rules those out for the product surface, and
the artwork follows the same constraint.
