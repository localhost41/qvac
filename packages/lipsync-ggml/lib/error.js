'use strict'

const { QvacErrorBase, addCodes } = require('@qvac/error')
const { name, version } = require('../package.json')

class QvacErrorAddonLipsync extends QvacErrorBase { }

// This library has error code range from 31001 to 32000
const ERR_CODES = Object.freeze({
  FAILED_TO_LOAD_WEIGHTS: 31001,
  FAILED_TO_DESTROY: 31002,
  MODEL_NOT_FOUND: 31003,
  INVALID_CONFIG: 31004,
  MISSING_REQUIRED_PARAMETER: 31005,
  INVALID_INPUT: 31006,
  JOB_ALREADY_RUNNING: 31007,
  INSTANCE_NOT_INITIALIZED: 31008,
  MODEL_UNLOADED: 31009,
  INFERENCE_FAILED: 31010
})

addCodes(
  {
    [ERR_CODES.FAILED_TO_LOAD_WEIGHTS]: {
      name: 'FAILED_TO_LOAD_WEIGHTS',
      message: message => `Failed to load weights, error: ${message}`
    },
    [ERR_CODES.FAILED_TO_DESTROY]: {
      name: 'FAILED_TO_DESTROY',
      message: message => `Failed to destroy instance, error: ${message}`
    },
    [ERR_CODES.MODEL_NOT_FOUND]: {
      name: 'MODEL_NOT_FOUND',
      message: path => `Lipsync GGUF not found: ${path}`
    },
    [ERR_CODES.INVALID_CONFIG]: {
      name: 'INVALID_CONFIG',
      message: message => `Invalid configuration: ${message}`
    },
    [ERR_CODES.MISSING_REQUIRED_PARAMETER]: {
      name: 'MISSING_REQUIRED_PARAMETER',
      message: paramName => `Missing required parameter: ${paramName}`
    },
    [ERR_CODES.INVALID_INPUT]: {
      name: 'INVALID_INPUT',
      message: message => `Invalid input: ${message}`
    },
    [ERR_CODES.JOB_ALREADY_RUNNING]: {
      name: 'JOB_ALREADY_RUNNING',
      message: () => 'Cannot set new job: a job is already set or being processed'
    },
    [ERR_CODES.INSTANCE_NOT_INITIALIZED]: {
      name: 'INSTANCE_NOT_INITIALIZED',
      message: () => 'Addon not initialized. Call load() first.'
    },
    [ERR_CODES.MODEL_UNLOADED]: {
      name: 'MODEL_UNLOADED',
      message: () => 'Model was unloaded'
    },
    [ERR_CODES.INFERENCE_FAILED]: {
      name: 'INFERENCE_FAILED',
      message: message => `Inference failed: ${message}`
    }
  },
  {
    name,
    version
  }
)

module.exports = {
  ERR_CODES,
  QvacErrorAddonLipsync
}
