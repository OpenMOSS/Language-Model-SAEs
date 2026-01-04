//  @ts-check

import { tanstackConfig } from '@tanstack/eslint-config'
import tseslint from 'typescript-eslint'

export default [
  {
    ignores: ['.output/**', 'dist/**'],
  },
  ...tanstackConfig,
  {
    plugins: {
      '@typescript-eslint': tseslint.plugin,
    },
    rules: {
      'no-shadow': 'off',
      '@typescript-eslint/no-unnecessary-condition': 'off',
      '@typescript-eslint/array-type': ['error', { default: 'array' }],
    },
  },
]
