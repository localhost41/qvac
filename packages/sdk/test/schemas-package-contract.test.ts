import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, it } from 'node:test'

type SchemaObject = {
  shape?: Record<string, { description?: unknown }>
}

type JsonSchema = {
  properties?: Record<string, JsonSchema>
  [key: string]: unknown
}

const packageRoot = fileURLToPath(new URL('..', import.meta.url))
const packageJson = JSON.parse(fs.readFileSync(path.join(packageRoot, 'package.json'), 'utf8')) as {
  name: string
  exports: Record<string, unknown>
}

function readJson(filePath: string): JsonSchema {
  return JSON.parse(fs.readFileSync(filePath, 'utf8')) as JsonSchema
}

function findObjectProperties(
  node: unknown,
  fieldNames: readonly string[]
): Array<Record<string, JsonSchema>> {
  if (node === null || typeof node !== 'object') return []

  const matches: Array<Record<string, JsonSchema>> = []
  const properties = (node as JsonSchema).properties
  if (
    properties &&
    Object.keys(properties).length === fieldNames.length &&
    fieldNames.every((fieldName) => fieldName in properties)
  ) {
    matches.push(properties)
  }

  for (const value of Object.values(node as Record<string, unknown>)) {
    matches.push(...findObjectProperties(value, fieldNames))
  }
  return matches
}

function assertDescribedFields(label: string, schema: SchemaObject): string[] {
  const fields = schema.shape ?? {}
  const fieldNames = Object.keys(fields)
  assert.ok(fieldNames.length > 0, `${label} should expose object fields`)

  for (const fieldName of fieldNames) {
    assert.equal(
      typeof fields[fieldName]?.description,
      'string',
      `${label}.${fieldName} should retain its public description`
    )
    assert.ok(
      (fields[fieldName]?.description as string).trim().length > 0,
      `${label}.${fieldName} should not have an empty description`
    )
  }
  return fieldNames
}

describe('@qvac/sdk 0.18 schemas package contract', () => {
  it('keeps the schemas subpath mapped to built runtime and declaration files', () => {
    const schemasExport = packageJson.exports['./schemas']
    assert.deepEqual(schemasExport, {
      types: './dist/src/schemas/public.d.ts',
      import: './dist/src/schemas/public.js',
      require: './dist/src/schemas/public.js'
    })

    for (const target of Object.values(schemasExport as Record<string, string>)) {
      assert.ok(
        fs.existsSync(path.join(packageRoot, target)),
        `package export target should exist after build: ${target}`
      )
    }

    const declarations = fs.readFileSync(
      path.join(packageRoot, 'dist/src/schemas/public.d.ts'),
      'utf8'
    )
    for (const name of [
      'llamacppCompletionConfigSchema',
      'llamacppEmbeddingConfigSchema',
      'modelSourceSchema',
      'LlamacppCompletionConfig',
      'LlamacppEmbeddingConfig',
      'ModelSource'
    ]) {
      assert.match(declarations, new RegExp(`\\b${name}\\b`), `${name} should be typed publicly`)
    }
  })

  it('exposes the documented aliases from the installed package subpath', async () => {
    const publicSchemas = (await import(`${packageJson.name}/schemas`)) as Record<string, unknown>
    assert.deepEqual(Object.keys(publicSchemas).sort(), [
      'llamacppCompletionConfigSchema',
      'llamacppEmbeddingConfigSchema',
      'modelSourceSchema'
    ])

    const completion = publicSchemas.llamacppCompletionConfigSchema as SchemaObject
    const embedding = publicSchemas.llamacppEmbeddingConfigSchema as SchemaObject
    const modelSource = publicSchemas.modelSourceSchema as {
      safeParse: (value: unknown) => { success: boolean }
    }
    const completionFields = assertDescribedFields('llamacppCompletionConfigSchema', completion)
    const embeddingFields = assertDescribedFields('llamacppEmbeddingConfigSchema', embedding)

    assert.equal(modelSource.safeParse('models/example.gguf').success, true)
    assert.equal(modelSource.safeParse({ src: 'models/example.gguf' }).success, true)

    const contract = readJson(path.join(packageRoot, 'contract', 'schema.json'))
    for (const [label, fields] of [
      ['completion', completionFields],
      ['embedding', embeddingFields]
    ] as const) {
      const matches = findObjectProperties(contract, fields)
      assert.ok(matches.length > 0, `contract/schema.json should contain the ${label} config`)
      for (const properties of matches) {
        for (const fieldName of fields) {
          assert.equal(
            properties[fieldName]?.description,
            (label === 'completion' ? completion : embedding).shape?.[fieldName]?.description,
            `contract description should match the public ${label} schema for ${fieldName}`
          )
        }
      }
    }
  })
})
